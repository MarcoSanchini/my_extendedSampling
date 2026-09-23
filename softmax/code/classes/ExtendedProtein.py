import attr
from attr import define
from copy import deepcopy

import torch
import torch.nn.functional as F

import custom_esm.utils.constants.esm3 as C


@define
class ExtendedProtein:
	sequence: str | None = None
	tokens: torch.Tensor | None = None
	logits: torch.Tensor | None = None

	am: torch.Tensor | None = None
	U_structure_ce: torch.Tensor | None = None
	#cm: torch.Tensor | None = None
	#plddt: torch.Tensor | None = None

	requires_grad: bool = True
	device: torch.device | str = "cpu"

	def __len__(self):
		if self.sequence is not None:
			return len(self.sequence)
		elif self.logits is not None:
			return len(self.logits)
		elif self.tokens is not None:
			return len(self.tokens) - 2
		else:
			raise ValueError(f"ExtendedProtein.__len__(): at least one field among (sequence, logits, tokens) in ExtendedProtein must be non-empty to access sequence length.")

	def expand(self, mask: torch.Tensor | None = None) -> None:
		#TODO: implement mask which determines which sites can be mutated
		used_tokens = torch.LongTensor([C.SEQUENCE_USED_VOCAB.index(amino) for amino in list(self.sequence)])
		self.logits = F.one_hot(
			used_tokens,
			num_classes=len(C.SEQUENCE_USED_VOCAB)
		).type(torch.float32).to(self.device)
		self.logits -= self.logits.mean(axis=-1, keepdim=True)
		self.logits.requires_grad = self.requires_grad

		self.tokens = torch.tensor(
			[C.SEQUENCE_BOS_TOKEN] +
			[C.SEQUENCE_VOCAB.index(amino) for amino in list(self.sequence)] +
			[C.SEQUENCE_EOS_TOKEN]
		).to(self.device)

	def collapse(self) -> None:
		used_aminos_list = [C.SEQUENCE_USED_VOCAB[used_token] for used_token in self.logits.argmax(-1)]
		self.sequence = "".join(used_aminos_list)
		self.tokens = torch.tensor(
			[C.SEQUENCE_BOS_TOKEN] +
			[C.SEQUENCE_VOCAB.index(used_amino) for used_amino in used_aminos_list] +
			[C.SEQUENCE_EOS_TOKEN]
		).to(self.device)

	def get_probs(self, T: float | None = None) -> torch.Tensor:
		if T is None:
			used_tokens = torch.LongTensor([C.SEQUENCE_USED_VOCAB.index(amino) for amino in list(self.sequence)])
			return F.one_hot(
				used_tokens,
				num_classes=len(C.SEQUENCE_USED_VOCAB)
			).type(torch.float32).to(self.device)
		else:
			return F.softmax(
				self.logits/T,
				dim=-1
			)

	def set_requires_grad(self, requires_grad: bool = True) -> None:
		self.requires_grad = requires_grad
		if self.logits is not None:
			self.logits.requires_grad = requires_grad

	def set_device(self, device: torch.device | str = "cpu") -> None:
		self.device = device
		if self.logits is not None:
			self.logits = self.logits.to(device)
		if self.tokens is not None:
			self.tokens = self.tokens.to(device)

	def save(self, dir, suffix="") -> None:
		for f in attr.fields(ExtendedProtein):
			if f.name in ["requires_grad", "device"]:
				continue
			torch.save(
				getattr(self, f.name).to("cpu") if f.name!="sequence" else getattr(self, f.name),
				f'{dir}/{f.name}{suffix}.pt',
			)

	def load(self, dir, suffix="") -> None:
		for f in attr.fields(ExtendedProtein):
			if f.name in ["requires_grad", "device"]:
				continue
			setattr(
				self,
				f.name,
				torch.load(f'{dir}/{f.name}{suffix}.pt')
			)
		self.tokens = self.tokens.to(self.device)
		self.logits = self.logits.to(self.device).detach()
		self.logits.requires_grad = self.requires_grad
		self.sequence = "".join([C.SEQUENCE_VOCAB[token] for token in list(self.tokens)[1:-1]])

	def clean(self) -> None:
		for f in attr.fields(ExtendedProtein):
			if f.name in ["requires_grad", "device"]:
				continue
			setattr(
				self,
				f.name,
				None
			)

	def copy(self):
		"""Create a deep copy of the ExtendedProtein instance."""
		return deepcopy(self)
