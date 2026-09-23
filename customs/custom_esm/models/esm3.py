from __future__ import annotations

import contextlib
from functools import partial
from typing import Callable

import attr
import einops
import torch
import torch.nn as nn
from attr import dataclass

from custom_esm.layers.regression_head import RegressionHead
from custom_esm.layers.transformer_stack import TransformerStack
from custom_esm.models.function_decoder import FunctionTokenDecoder
from custom_esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from custom_esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinTensor,
    ForwardAndSampleOutput,
    ForwardTrackData,
    GenerationConfig,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
    SamplingConfig,
)
from custom_esm.tokenization import TokenizerCollectionProtocol
from custom_esm.utils import encoding
from custom_esm.utils.constants import esm3 as C
from custom_esm.utils.constants.models import (
    ESM3_OPEN_SMALL,
    normalize_model_name,
)
from custom_esm.utils.decoding import decode_protein_tensor
from custom_esm.utils.generation import (
    _batch_forward,
    _sample_per_prompt,
    _slice_tensor_dataclass,
    iterative_sampling_raw,
    iterative_sampling_tokens,
)
from custom_esm.utils.misc import rbf
from custom_esm.utils.sampling import (
    _BatchedESMProteinTensor,
    get_default_sampling_config,
    validate_sampling_config,
)
from custom_esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)

# ADDITIONAL LIBRARIES
import torch.nn.functional as F
from custom_esm.utils.custom import clean_and_symm
from custom_esm.utils.decoding import decode_sequence, custom_decode_structure


@dataclass
class ESMOutput:
    sequence_logits: torch.Tensor
    structure_logits: torch.Tensor
    secondary_structure_logits: torch.Tensor
    sasa_logits: torch.Tensor
    function_logits: torch.Tensor
    residue_logits: torch.Tensor
    embeddings: torch.Tensor


class EncodeInputs(nn.Module):
    """
    Module for encoding input features in the ESM-3 model.

    Args:
        d_model (int): The dimensionality of the model's hidden states.
    """

    def __init__(self, d_model: int):
        super().__init__()

        # Sequence
        self.sequence_embed = nn.Embedding(64, d_model)
        # Mandatory information
        self.plddt_projection = nn.Linear(16, d_model)
        self.structure_per_res_plddt_projection = nn.Linear(16, d_model)

        # Structure
        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)

        # "Structural" features
        self.ss8_embed = nn.Embedding(8 + 3, d_model)
        self.sasa_embed = nn.Embedding(16 + 3, d_model)

        # "Functional" features
        self.function_embed = nn.ModuleList(
            [nn.Embedding(260, d_model // 8, padding_idx=0) for _ in range(8)]
        )

        self.residue_embed = nn.EmbeddingBag(1478, d_model, mode="sum", padding_idx=0)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        average_plddt: torch.Tensor,
        per_res_plddt: torch.Tensor,
        ss8_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
        residue_annotation_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embed = self.sequence_embed(sequence_tokens)

        rbf_16_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        # the `masked_fill(padding_mask.unsqueeze(2), 0)` for the two below is unnecessary
        # as pad tokens never even interact with the "real" tokens (due to sequence_id)
        plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = self.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt)
        )

        # Structure + "structural features" embeds
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)

        # "Functional" features embeds
        function_embed = torch.cat(
            [
                embed_fn(funcs)
                for embed_fn, funcs in zip(
                    self.function_embed, function_tokens.unbind(-1)
                )
            ],
            -1,
        )

        # Residue embeds
        B, L, N = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(
            einops.rearrange(
                residue_annotation_tokens, "B L N -> (B L) N", B=B, L=L, N=N
            )
        )
        residue_embed = einops.rearrange(residue_embed, "(B L) D -> B L D", B=B, L=L)

        return (
            sequence_embed
            + plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )

    """ ############################################################################ """
    """ ############################## CUSTOM METHODS ############################## """
    """ ############################################################################ """
    def format_sequence_probs(self, probs):
        prepend_to = C.SEQUENCE_VOCAB.index(C.SEQUENCE_USED_VOCAB[0])
        append_from = C.SEQUENCE_VOCAB.index(C.SEQUENCE_USED_VOCAB[-1])
        prepend_tokens = C.SEQUENCE_VOCAB[:prepend_to]
        append_tokens = C.SEQUENCE_VOCAB[append_from+1:]

        B, L, d = probs.shape
        prepend_probs = torch.zeros((B, L, len(prepend_tokens))).to(probs)
        append_probs = torch.zeros((B, L, len(append_tokens)+64-len(C.SEQUENCE_VOCAB))).to(probs)
        probs = torch.concat(
            (prepend_probs, probs, append_probs),
            axis=-1
        )

        bos_probs = F.one_hot(torch.LongTensor([C.SEQUENCE_BOS_TOKEN]), num_classes=64).to(probs).repeat(B, 1, 1)
        eos_probs = F.one_hot(torch.LongTensor([C.SEQUENCE_EOS_TOKEN]), num_classes=64).to(probs).repeat(B, 1, 1)
        probs = torch.concat(
            (bos_probs, probs, eos_probs),
            axis=1
        )
        return probs

    def manual_embed(self, x):
        x = torch.tensordot(x, self.sequence_embed.weight, dims=([2], [0]))
        return x

    def custom_forward(
        self,
        sequence_probs: torch.Tensor,
        structure_tokens: torch.Tensor,
        average_plddt: torch.Tensor,
        per_res_plddt: torch.Tensor,
        ss8_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
        residue_annotation_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embed = self.manual_embed(
                self.format_sequence_probs(sequence_probs)
        )

        rbf_16_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        # the `masked_fill(padding_mask.unsqueeze(2), 0)` for the two below is unnecessary
        # as pad tokens never even interact with the "real" tokens (due to sequence_id)
        plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = self.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt)
        )

        # Structure + "structural features" embeds
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)

        # "Functional" features embeds
        function_embed = torch.cat(
            [
                embed_fn(funcs)
                for embed_fn, funcs in zip(
                    self.function_embed, function_tokens.unbind(-1)
                )
            ],
            -1,
        )

        # Residue embeds
        B, L, N = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(
            einops.rearrange(
                residue_annotation_tokens, "B L N -> (B L) N", B=B, L=L, N=N
            )
        )
        residue_embed = einops.rearrange(residue_embed, "(B L) D -> B L D", B=B, L=L)

        return (
            sequence_embed
            + plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )


class OutputHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 64)
        self.structure_head = RegressionHead(d_model, 4096)
        self.ss8_head = RegressionHead(d_model, 8 + 3)
        self.sasa_head = RegressionHead(d_model, 16 + 3)
        self.function_head = RegressionHead(d_model, 260 * 8)
        self.residue_head = RegressionHead(d_model, 1478)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> ESMOutput:
        sequence_logits = self.sequence_head(x)
        structure_logits = self.structure_head(x)
        secondary_structure_logits = self.ss8_head(x)
        sasa_logits = self.sasa_head(x)
        function_logits = self.function_head(x)
        function_logits = einops.rearrange(function_logits, "... (k v) -> ... k v", k=8)

        residue_logits = self.residue_head(x)

        return ESMOutput(
            sequence_logits=sequence_logits,
            structure_logits=structure_logits,
            secondary_structure_logits=secondary_structure_logits,
            sasa_logits=sasa_logits,
            function_logits=function_logits,
            residue_logits=residue_logits,
            embeddings=embed,
        )


class ESM3(nn.Module, ESM3InferenceClient):
    """
    ESM3 model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        v_heads (int): The number of attention heads in the variational transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int,
        n_layers: int,
        structure_encoder_fn: Callable[[torch.device | str], StructureTokenEncoder],
        structure_decoder_fn: Callable[[torch.device | str], StructureTokenDecoder],
        function_decoder_fn: Callable[[torch.device | str], FunctionTokenDecoder],
        tokenizers: TokenizerCollectionProtocol,
    ):
        super().__init__()
        self.encoder = EncodeInputs(d_model)
        self.transformer = TransformerStack(
            d_model, n_heads, v_heads, n_layers, mask_and_zero_frameless=True
        )
        self.output_heads = OutputHeads(d_model)

        self.structure_encoder_fn = structure_encoder_fn
        self.structure_decoder_fn = structure_decoder_fn
        self.function_decoder_fn = function_decoder_fn

        self._structure_encoder = None
        self._structure_decoder = None
        self._function_decoder = None

        self.tokenizers = tokenizers

    @classmethod
    def from_pretrained(
        cls, model_name: str = ESM3_OPEN_SMALL, device: torch.device | None = None
    ) -> ESM3:
        from custom_esm.pretrained import load_local_model

        model_name = normalize_model_name(model_name)
        if not model_name:
            raise ValueError(f"Model name {model_name} is not a valid ESM3 model name.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_local_model(model_name, device=device)
        if device.type != "cpu":
            model = model.to(torch.bfloat16)
        assert isinstance(model, ESM3)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def raw_model(self):
        return self

    def get_structure_encoder(self) -> StructureTokenEncoder:
        if self._structure_encoder is None:
            self._structure_encoder = self.structure_encoder_fn(self.device)
        return self._structure_encoder

    def get_structure_decoder(self) -> StructureTokenDecoder:
        if self._structure_decoder is None:
            self._structure_decoder = self.structure_decoder_fn(self.device)
        return self._structure_decoder

    def get_function_decoder(self) -> FunctionTokenDecoder:
        if self._function_decoder is None:
            self._function_decoder = self.function_decoder_fn(self.device)
        return self._function_decoder

    def forward(
        self,
        *,
        sequence_tokens: torch.Tensor | None = None,
        structure_tokens: torch.Tensor | None = None,
        ss8_tokens: torch.Tensor | None = None,
        sasa_tokens: torch.Tensor | None = None,
        function_tokens: torch.Tensor | None = None,
        residue_annotation_tokens: torch.Tensor | None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ) -> ESMOutput:
        """
        Performs forward pass through the ESM3 model. Check utils to see how to tokenize inputs from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            structure_tokens (torch.Tensor, optional): The structure tokens.
            ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
            sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
            function_tokens (torch.Tensor, optional): The function tokens.
            residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
            average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
            per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
                otherwise, use average_plddt.
            structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
            chain_id (torch.Tensor, optional): The chain ID
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMOutput: The output of the ESM3 model.

        Raises:
            ValueError: If at least one of the inputs is None.

        """
        # Reasonable defaults:
        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        x, embedding, _ = self.transformer(
            x, sequence_id, affine, affine_mask, chain_id
        )
        return self.output_heads(x, embedding)

    # The following methods are for the ESM3InferenceClient interface
    def generate(self, input: ProteinType, config: GenerationConfig) -> ProteinType:
        """Wrap around batched generation."""
        proteins = self.batch_generate([input], [config])
        assert len(proteins) == 1
        return proteins[0]

    def batch_generate(
            self, inputs: list[ProteinType], configs: list[GenerationConfig]
    ) -> list[ProteinType]:
        assert len(inputs) == len(
            configs
        ), "Must have the same number of prompts and configs."

        if inputs == []:
            # Nothing to do.
            return []

        # Make sure prompts are of the same type.
        t = type(inputs[0])
        for i in range(1, len(inputs)):
            assert isinstance(inputs[i], t), (
                "Prompts must have the same type. Got "
                f"{t.__name__ and type(inputs[i]).__name__} instead."
            )

        if isinstance(inputs[0], ESMProtein):
            return iterative_sampling_raw(self, inputs, configs)  # type: ignore
        elif isinstance(inputs[0], ESMProteinTensor):
            return iterative_sampling_tokens(
                self,
                inputs,  # type: ignore
                configs,
                self.tokenizers,  # type: ignore
            )
        else:
            raise ValueError("Input must be an ESMProtein or ESMProteinTensor")

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        input = attr.evolve(input)  # Make a copy

        sequence_tokens = None
        structure_tokens = None
        secondary_structure_tokens = None
        sasa_tokens = None
        function_tokens = None
        residue_annotation_tokens = None

        coordinates = None

        if input.sequence is not None:
            sequence_tokens = encoding.tokenize_sequence(
                input.sequence, self.tokenizers.sequence, add_special_tokens=True
            )
        if input.secondary_structure is not None:
            secondary_structure_tokens = encoding.tokenize_secondary_structure(
                input.secondary_structure,
                self.tokenizers.secondary_structure,
                add_special_tokens=True,
            )
        if input.sasa is not None:
            sasa_tokens = encoding.tokenize_sasa(
                input.sasa, self.tokenizers.sasa, add_special_tokens=True
            )

        # Infer input length
        sequence_length = -1
        if sequence_tokens is not None:
            sequence_length = len(sequence_tokens)
        elif secondary_structure_tokens is not None:
            sequence_length = len(secondary_structure_tokens)
        elif sasa_tokens is not None:
            sequence_length = len(sasa_tokens)

        # Try to infer input length from structure data
        if input.coordinates is not None:
            coordinates, _, structure_tokens = encoding.tokenize_structure(
                input.coordinates,
                self.get_structure_encoder(),
                structure_tokenizer=self.tokenizers.structure,
                reference_sequence=input.sequence or "",
                add_special_tokens=True,
            )
            if sequence_length == -1:
                sequence_length = len(structure_tokens)

        if sequence_length == -1:
            raise ValueError(
                "Cannot infer input length from input data. Please provide one of: sequence, structure, secondary_structure, sasa.\n"
                "To condition on sequence length only, use ESM3LocalInferenceClient.get_default_sequence(sequence_length) to generate a default sequence input."
            )

        # Function and Residue annotations
        if input.function_annotations is not None:
            if input.sequence is None:
                reference_sequence = encoding.get_default_sequence(sequence_length - 2)
            else:
                reference_sequence = input.sequence
            (function_tokens, residue_annotation_tokens) = (
                encoding.tokenize_function_annotations(
                    input.function_annotations,
                    reference_sequence=reference_sequence,
                    function_tokenizer=self.tokenizers.function,
                    residue_annotation_tokenizer=self.tokenizers.residue_annotations,
                    add_special_tokens=True,
                )
            )

        return ESMProteinTensor(
            sequence=sequence_tokens,
            structure=structure_tokens,
            secondary_structure=secondary_structure_tokens,
            sasa=sasa_tokens,
            function=function_tokens,
            residue_annotations=residue_annotation_tokens,
            coordinates=coordinates,
        ).to(next(self.parameters()).device)

    def decode(self, input: ESMProteinTensor) -> ESMProtein:
        return decode_protein_tensor(
            input=input,
            tokenizers=self.tokenizers,
            structure_token_decoder=self.get_structure_decoder(),
            function_token_decoder=self.get_function_decoder(),
        )

    def logits(
        self,
        input: ESMProteinTensor | _BatchedESMProteinTensor,
        config: LogitsConfig = LogitsConfig(),
    ) -> LogitsOutput:
        if not isinstance(input, _BatchedESMProteinTensor):
            # Create batch dimension if necessary.
            input = _BatchedESMProteinTensor.from_protein_tensor(input)

        device = torch.device(input.device)

        # Default plddt conditioning for inference. 1s where coordinates are provided.
        if input.coordinates is None:
            per_res_plddt = None
        else:
            # 1.0 if all coordinates at specific indices have valid non-nan values.
            per_res_plddt = input.coordinates.isfinite().all(dim=-1).any(dim=-1).float()

        with (
            torch.no_grad(),  # Assume no gradients for now...
            torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16)  # type: ignore
            if device.type == "cuda"
            else contextlib.nullcontext(),
        ):
            output = self.forward(
                sequence_tokens=input.sequence,
                structure_tokens=input.structure,
                ss8_tokens=input.secondary_structure,
                sasa_tokens=input.sasa,
                function_tokens=input.function,
                residue_annotation_tokens=input.residue_annotations,
                average_plddt=torch.tensor(1.0, device=input.device),
                per_res_plddt=per_res_plddt,
                structure_coords=input.coordinates,
                chain_id=None,
                sequence_id=None,
            )

        output = ESMOutput(
            **{k: v.to(device).to(torch.float32) for k, v in vars(output).items()}
        )

        return LogitsOutput(
            logits=ForwardTrackData(
                sequence=output.sequence_logits if config.sequence else None,
                structure=output.structure_logits if config.structure else None,
                secondary_structure=output.secondary_structure_logits
                if config.secondary_structure
                else None,
                sasa=output.sasa_logits if config.sasa else None,
                function=output.function_logits if config.function else None,
            ),
            residue_annotation_logits=output.residue_logits
            if config.residue_annotations
            else None,
            embeddings=output.embeddings if config.return_embeddings else None,
        )

    def forward_and_sample(
        self, input: ESMProteinTensor, sampling_configuration: SamplingConfig
    ) -> ForwardAndSampleOutput:
        validate_sampling_config(sampling_configuration, on_invalid="warn")

        protein_tensor = attr.evolve(input)  # Make a copy

        device = next(self.parameters()).device

        sampling_config = sampling_configuration
        if sampling_config is None:
            sampling_config = get_default_sampling_config(self.tokenizers)

        # Initialize default values for missing tracks
        default_protein_tensor = ESMProteinTensor.empty(
            len(input) - 2, tokenizers=self.tokenizers, device=input.device
        )
        for track in attr.fields(ESMProteinTensor):
            if getattr(protein_tensor, track.name, None) is None:
                setattr(
                    protein_tensor,
                    track.name,
                    getattr(default_protein_tensor, track.name, None),
                )

        if len(protein_tensor) <= 0:
            raise ValueError("No input data provided")

        # Move input protein to proper device.
        batched_protein = _BatchedESMProteinTensor.from_protein_tensor(protein_tensor)
        batched_protein.to(device)

        logits_output: LogitsOutput = _batch_forward(self, batched_protein)
        forward_and_sample_out: ForwardAndSampleOutput = _sample_per_prompt(
            batched_protein, logits_output, sampling_config, self.tokenizers
        )

        # There is only 1 prompt to sample for.
        return _slice_tensor_dataclass(forward_and_sample_out, 0)


    """ ############################################################################ """
    """ ############################## CUSTOM METHODS ############################## """
    """ ############################################################################ """
    def _default(
            self,
            L: int,
            device: torch.device,
    ):
        t = self.tokenizers
        defaults = lambda tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device)
        )
        ss8_tokens = defaults(C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(C.SASA_PAD_TOKEN)
        average_plddt = defaults(1).float()
        per_res_plddt = defaults(0).float()
        chain_id = defaults(0)

        residue_annotation_tokens = torch.full(
            (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
        )

        function_tokens = torch.full(
            (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
        )

        structure_coords = torch.full(
            (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
        )
        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(C.STRUCTURE_MASK_TOKEN)
        structure_tokens[0, 0] = C.STRUCTURE_BOS_TOKEN      # Assuming: one sequence, no padding
        structure_tokens[0, L-1] = C.STRUCTURE_EOS_TOKEN    #

        return (
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        ), affine, affine_mask

    def _last_attention(
            self,
            sequence_probs: torch.Tensor,
            chain_id: torch.Tensor | None = None,
            sequence_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Performs forward pass through the ESM3 model, limited up to the TransformerStack.

        Args:
            sequence_probs (torch.Tensor): The amino acid probability distribution.
            chain_id (torch.Tensor, optional): The chain ID.
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            attention: The last attention tensor (as mean among the different heads) of shape (batch_size, sequence_length, sequence_length).
        """
        # Check sequence_probs dimensions
        if sequence_probs.ndim == 2:
            sequence_probs.unsqueeze(0)
        elif sequence_probs.ndim == 3:
            assert sequence_probs.shape[0] == 1, "Attention can be computed only for single sequences (multi-sequence prediction not yet implemented)."
        else:
            raise ValueError(f"Unexpected number of dimensions for sequence_probs. Found {sequence_probs.ndim}, expected 2 or 3.")

        # Extract L and device
        L, device = sequence_probs.shape[1], sequence_probs.device
        L += 2 # adding BOS and EOS tokens

        # Remaining (default) tokens and affine/affine_mask
        default_tokens, affine, affine_mask = self._default(L, device)
        
        x = self.encoder.custom_forward(
            sequence_probs,
            *default_tokens,
        )
        attention = self.transformer.last_attention(
            x, sequence_id, affine, affine_mask, chain_id
        )
        return attention

    def predict_attention(
            self,
            sequence_probs: torch.Tensor,
            chain_id: torch.Tensor | None = None,
            sequence_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with (
                torch.autocast(enabled=True, device_type=sequence_probs.device.type, dtype=torch.bfloat16)
                if sequence_probs.device.type == "cuda"
                else contextlib.nullcontext(),
        ):
            attention = self._last_attention(
                    sequence_probs=sequence_probs.unsqueeze(0),
                    chain_id=chain_id,
                    sequence_id=sequence_id
            )
        return clean_and_symm(attention).squeeze(0)


    def _structure_logits(
            self,
            sequence_probs: torch.Tensor,
            chain_id: torch.Tensor | None = None,
            sequence_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the ESM3 model, using the FULL
        TransformerStack (not just up to the last attention, cf
        _last_attention above) to obtain structure_logits: the model's
        per-residue prediction over the 4096-token discrete structure
        vocabulary. Differentiable w.r.t. sequence_probs, same as
        _last_attention/predict_attention.

        Args:
            sequence_probs (torch.Tensor): The amino acid probability distribution.
            chain_id (torch.Tensor, optional): The chain ID.
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            structure_logits: shape (batch_size, sequence_length, 4096),
            including the BOS/EOS positions (caller slices them off).
        """
        # Check sequence_probs dimensions
        if sequence_probs.ndim == 2:
            sequence_probs.unsqueeze(0)
        elif sequence_probs.ndim == 3:
            assert sequence_probs.shape[0] == 1, "Structure logits can be computed only for single sequences (multi-sequence prediction not yet implemented)."
        else:
            raise ValueError(f"Unexpected number of dimensions for sequence_probs. Found {sequence_probs.ndim}, expected 2 or 3.")

        # Extract L and device
        L, device = sequence_probs.shape[1], sequence_probs.device
        L += 2 # adding BOS and EOS tokens

        # Remaining (default) tokens and affine/affine_mask
        default_tokens, affine, affine_mask = self._default(L, device)

        x = self.encoder.custom_forward(
            sequence_probs,
            *default_tokens,
        )
        # Full stack forward (not last_attention): need the final, post-norm
        # hidden state to feed the structure head. TransformerStack.forward
        # returns (post_norm, pre_norm, hiddens); post_norm is what every
        # OutputHeads projection (structure included) is meant to consume.
        x, _, _ = self.transformer(
            x, sequence_id, affine, affine_mask, chain_id
        )
        return self.output_heads.structure_head(x)


    def predict_structure_logits(
            self,
            sequence_probs: torch.Tensor,
            chain_id: torch.Tensor | None = None,
            sequence_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with (
                torch.autocast(enabled=True, device_type=sequence_probs.device.type, dtype=torch.bfloat16)
                if sequence_probs.device.type == "cuda"
                else contextlib.nullcontext(),
        ):
            structure_logits = self._structure_logits(
                    sequence_probs=sequence_probs.unsqueeze(0),
                    chain_id=chain_id,
                    sequence_id=sequence_id
            )
        return structure_logits.squeeze(0)


    def custom_decode(
            self,
            tensor: ESMProteinTensor,
            infer_cbeta: bool = True,
    ) -> ESMProtein:
        assert tensor.sequence is not None, "custom_decode(): found tensor.sequence=None, sequence tokens are necessary for structure decoding."
        assert tensor.structure is not None, "custom_decode(): found tensor.structure=None, nothing to decode."
        sequence = decode_sequence(tensor.sequence, self.tokenizers.sequence)
        coordinates, plddt, ptm = custom_decode_structure(
            structure_tokens=tensor.structure,
            structure_decoder=self.get_structure_decoder(),
            structure_tokenizer=self.tokenizers.structure,
            sequence=sequence,
            infer_cbeta=infer_cbeta,
        )
        return ESMProtein(
                sequence=sequence,
                coordinates=coordinates,
                plddt=plddt,
                ptm=ptm,
        )

    def predict_protein(
            self,
            tensor: ESMProteinTensor,
            config: GenerationConfig,
            infer_cbeta: bool = True,
    ) -> ESMProtein:
        output = self.generate(input=tensor, config=config)
        return self.custom_decode(output, infer_cbeta=infer_cbeta)


    """ #################################################################################### """
    """ ############################## CUSTOM METHODS - EXTRA ############################## """
    """ #################################################################################### """
    def layers_attention(
            self,
            sequence_probs: torch.Tensor,
            layers: list[int] | None = None,
            chain_id: torch.Tensor | None = None,
            sequence_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Performs forward pass through the ESM3 model, limited up to the TransformerStack.

        Args:
            sequence_probs (torch.Tensor): The amino acid probability distribution.
            layers (list[int], optional): The layers from which attention must be explicitly computed (if None, attention is computed for every layer).
            chain_id (torch.Tensor, optional): The chain ID.
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            attentions: The list of attention tensors (as mean among the different heads) for each layer in layers, each of shape (batch_size, sequence_length, sequence_length).
        """
        # Extract L and device
        if sequence_probs.ndim == 2:
            sequence_probs.unsqueeze(0)
        elif sequence_probs.ndim == 3:
            assert sequence_probs.shape[0] == 1, "Attention can be computed only for single sequences (multi-sequence prediction not yet implemented)."
        else:
            raise ValueError(f"Unexpected number of dimensions for sequence_probs. Found {sequence_probs.ndim}, expected 2 or 3.")
        
        L, device = sequence_probs.shape[1], sequence_probs.device
        L += 2 # adding BOS and EOS tokens

        # Remaining (default) tokens and affine/affine_mask
        default_tokens, affine, affine_mask = self._default(L, device)
        if layers is None:
            layers = torch.arange(len(self.transformer.blocks)).tolist()
        
        x = self.encoder.custom_forward(
            sequence_probs,
            *default_tokens,
        )
        attentions = self.transformer.layers_attention(
            x, layers, sequence_id, affine, affine_mask, chain_id
        )
        return [clean_and_symm(attention) for attention in attentions]
