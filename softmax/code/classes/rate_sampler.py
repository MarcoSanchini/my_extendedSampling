import os
from io import StringIO
from time import process_time as ptime

import torch
import math

# CUSTOM ESM
#from huggingface_hub import login
from custom_esm.models.esm3 import ESM3
from custom_esm.sdk.api import ESM3InferenceClient
import custom_esm.utils.constants.esm3 as C

# SAMPLING
import sys
sys.path.append('..')
from .ExtendedProtein import ExtendedProtein
from generator.custom_generator import CustomGenerator
from utils.general import create_path
from utils.predictions import init_structure_config
from utils.energies import compute_U_am, compute_U_structure_ce, compute_entropy
from utils.operations import compute_Hd, is_subset, merge_dict
from utils.proposals import log_pointing_prob



# Non-canonical / ambiguous residue codes at the tail of SEQUENCE_USED_VOCAB
# (X=unknown, B=Asx, Z=Glx, U=selenocysteine, O=pyrrolysine). Never valid
# proposal targets, same as the site's own current amino acid isn't once
# it's been chosen to change (see class docstring): both are excluded from
# the competition by SLICING them out of the candidate set entirely (see
# _competition_indices below), not by penalizing them in a fixed-size
# tensor. That's a deliberate change from an earlier version, which used a
# large-but-finite additive penalty (-1e6) for both exclusions: correct,
# but it left a latent risk that step_coef*grad (which grows with dt^2)
# could eventually approach that constant's scale at large dt and weaken
# the exclusion. Slicing removes that risk entirely -- excluded classes
# are structurally absent from the tensor being maximized over, not just
# very unlikely to win.
_NONCANONICAL_RESIDUES = ('X', 'B', 'U', 'Z', 'O')



# -------------------------------------------------------------------------- #
# Locally-Balanced Rate ExtendedProtein Sampler (single-site by default)     #
#                                                                             #
# Each move: (1) picks n_sites distinct sites uniformly at random (pars key   #
# n_sites, default 1, which is the single-site move this sampler started     #
# from). The choice is symmetric and state-independent, so it cancels exactly #
# out of the Metropolis-Hastings ratio and needs no correction term; (2) at   #
# those sites only, draws a gradient-informed displacement and proposes a     #
# substitution at each one EXCLUDING its current amino acid from the          #
# competition (product over k != i,j, the original per-site formula this      #
# sampler started from -- once a site is chosen to change, "staying" is not a #
# candidate any more, only which OTHER amino acid to move to is); (3)         #
# accepts/rejects with an exact Metropolis-Hastings correction using the true #
# (recomputed) energy of the candidate and the true reverse-proposal          #
# probability, so the target distribution is exp(-U/T) exactly, not merely   #
# to first order.                                                             #
#                                                                             #
# This replaces an earlier version of this sampler that proposed all sites   #
# jointly every move (each site voting to stay or change via a K-way         #
# competition that included "stay"). That design was found, empirically, to  #
# touch ~all sites simultaneously regardless of dt/T (the K-way-including-   #
# stay competition has no small-dt limit that favors staying -- see          #
# ../../DEVLOG.txt), and U_am is strongly non-separable across simultaneously #
# -changed sites (test_joint_coupling.py measured coupling terms of the same #
# order as, or larger than, the individual site effects), so joint moves     #
# produced dU in the hundreds and were essentially always rejected. Fixing   #
# "how many sites change" to exactly one per move removes that failure mode  #
# by construction: with only one site changing, there is nothing for it to   #
# couple with. See DEVLOG.txt for the full history. n_sites > 1 brings joint  #
# moves back on purpose (2026-10-15): the coupling is then handled by the     #
# acceptance test rather than avoided by construction.                        #
#                                                                             #
# The gradient used for the chosen sites is computed by relaxing ONLY those  #
# sites (softmax(logits/T_sftm), differentiable) while every OTHER site is    #
# held at its exact discrete (hard one-hot) value -- not a whole-sequence    #
# softened relaxation. This costs the same (attention/embedding cost does    #
# not depend on how many positions are "soft" vs "hard") and is a more       #
# faithful partial derivative: it matches the exact discrete context used    #
# everywhere else (accept/reject, the reverse-probability pass).            #
#                                                                             #
# Per move: 2 backward passes (grad at A, grad at B, each over the relaxed   #
# chosen site(s)) + 1 forward pass (exact U at B) through ESM3. The cost does #
# not grow with n_sites: one relaxed pass covers all of them.                #
#                                                                             #
# The total energy U is lambda_am*U_am + lambda_structure_ce*U_structure_ce  #
# + lambda_S*entropy. U_am (utils/energies.py:compute_U_am) is a squared     #
# log-space distance between attention maps. U_structure_ce                  #
# (compute_U_structure_ce) is a cross-entropy between a candidate's          #
# predicted structure-token distribution and the reference's own (argmax)   #
# structure tokens -- closer in spirit to Zambon et al 2024's contact-map    #
# effective energy than U_am is, at the cost of needing its own separate    #
# characterization (it does NOT share U_am's provable U(ref)=0 floor: cross- #
# entropy against a discrete target has no such guarantee). Both terms come  #
# from the SAME differentiable ESM3 forward pass on soft sequence_probs      #
# (custom_esm/models/esm3.py:predict_attention / predict_structure_logits),  #
# so both remain usable in the gradient-informed proposal. lambda_am is the  #
# only one enabled by default (lambda_structure_ce defaults to 0.0); set     #
# either or both to compare how they behave, per DEVLOG.txt.                #
# -------------------------------------------------------------------------- #
class ExtendedProteinRateSampler():

	def __init__(
			self,
			config_settings: dict,
			name: str = 'ExtendedProteinRateSampler',
	):
		self.config_settings = config_settings
		self.config = init_structure_config(**self.config_settings)
		self.name = name

		# The next commented line is necessary only for the first login
		#login()
		self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open")
		for param in self.model.parameters():
			param.requires_grad = False

		self._init_attributes()



	def sample(
			self,
			ref_seq: str,
			pars: dict,
			settings: dict | None = None,
	):
		eprot, data, pars, settings = self._setup(ref_seq, pars, settings)
		accepted_total = data["acc_moves"]

		for move in range(data["move"]+1, pars["moves"]+1):

			eprot, info = self._step(eprot, data["U"], pars)

			if info["accepted"]:
				accepted_total += 1
				data["U"] = info["U_B"]
				data["U_am"] = info["U_am_B"]
				data["U_structure_ce"] = info["U_structure_ce_B"]
				data["Hd_to_ref"] = compute_Hd(eprot.logits, self.ref_eprot.logits)

			data["move"] = move
			data["site"] = info["site"]
			data["cur_aa"] = info["cur_aa"]
			data["proposed_aa"] = info["proposed_aa"]
			data["dU"] = info["dU"]
			data["log_a_AB"] = info["log_a_AB"]
			data["log_a_BA"] = info["log_a_BA"]
			data["log_ratio"] = info["log_ratio"]
			data["accepted"] = info["accepted"]
			data["acc_moves"] = accepted_total
			data["acc_rate"] = accepted_total / move

			data["time"] = ptime() - self.t0

			self._extend_buffer(data)

			if move%settings['log_step'] == 0:
				self._save_log(eprot, data, settings)
			if move%settings['print_step'] == 0:
				self._print_status(data)

		del self.ref_eprot, self.log, self.generator, self.t0



	# ------------------------------------------------------------------ #
	# One Monte Carlo move (see the class header). Picks n_sites sites,  #
	# proposes a gradient-informed substitution at each, and accepts or  #
	# rejects the joint change with an exact Metropolis-Hastings ratio.  #
	# Returns (eprot, info): eprot is the accepted candidate or the      #
	# unchanged input. With n_sites=1 it makes the same random draws, in #
	# the same order, as the single-site move it replaced.               #
	# ------------------------------------------------------------------ #
	def _step(self, eprot: ExtendedProtein, U_A: float, pars: dict):
		L = len(eprot.sequence)
		sites = self._draw_sites(L, pars['n_sites'])

		grad_A = self._grad_pass_sites(eprot, sites, pars)
		cur_idx = [int(eprot.logits[site].argmax(dim=-1).item()) for site in sites]

		sigma = pars['dt'] * math.sqrt(pars['T']/pars['M'])
		step_coef = pars['dt']**2. / (2.*pars['M'])
		mu_A = -step_coef*grad_A

		tgt_idx, log_a_AB = self._draw_targets(mu_A, cur_idx, sigma, pars)

		eprot_B = self._substitute(eprot, sites, tgt_idx)
		grad_B = self._grad_pass_sites(eprot_B, sites, pars)
		mu_B = -step_coef*grad_B
		log_a_BA = self._reverse_logprob(mu_B, cur_idx, tgt_idx, sigma, pars)

		U_B, U_am_B, eprot_B = self._exact_energy(eprot_B, pars)

		dU = U_B - U_A
		log_ratio = -dU/pars['T'] + (log_a_BA - log_a_AB)

		u = torch.rand(1, device=self.generator.device, generator=self.generator.get()).item()
		accepted = math.log(max(u, 1e-300)) <= min(0., log_ratio)

		vocab = C.SEQUENCE_USED_VOCAB
		info = {
			"site": ",".join(str(site) for site in sites),
			"cur_aa": ",".join(vocab[idx] for idx in cur_idx),
			"proposed_aa": ",".join(vocab[idx] for idx in tgt_idx),
			"proposed_sequence": eprot_B.sequence,
			"U_B": U_B, "U_am_B": U_am_B, "U_structure_ce_B": eprot_B.U_structure_ce.item(),
			"dU": dU, "log_a_AB": log_a_AB, "log_a_BA": log_a_BA, "log_ratio": log_ratio,
			"accepted": int(accepted),
		}

		return (eprot_B, info) if accepted else (eprot, info)


	# n distinct sites, uniform over subsets of size n. For n=1 this is the
	# single draw of the single-site sampler, so n_sites=1 replays old runs.
	def _draw_sites(self, L: int, n: int) -> list:
		sites = []
		while len(sites) < n:
			site = int(torch.randint(0, L, (1,), generator=self.generator.get(), device=self.generator.device).item())
			if site not in sites:
				sites.append(site)
		return sites


	# Gradient-informed targets, one per chosen site. At each site the
	# displacement dx_k = mu_k + sigma*xi_k (xi iid standard normal, from the
	# momentum draw) is maximized over that site's competitors, and the argmax
	# is the target. log_a is the exact log-probability of the realized
	# targets (log_pointing_prob); the joint proposal is the product over sites.
	def _draw_targets(self, mu: torch.Tensor, cur_idx: list, sigma: float, pars: dict):
		momentum = self._extract_momenta(mu.shape, pars['T'], pars['M'])
		delta_x = pars['dt']*momentum/pars['M'] + mu
		delta_x = delta_x - delta_x.mean(dim=-1, keepdim=True)   # gauge-fix over the FULL space, before any slicing

		tgt_idx, log_a = [], 0.
		for site, cur in enumerate(cur_idx):
			competitors = self._competition_indices(cur)
			local_tgt = delta_x[site][competitors].argmax(dim=-1)   # index into competitors
			tgt_idx.append(int(competitors[local_tgt].item()))      # global vocab index, never cur
			log_a += log_pointing_prob(mu[site][competitors], sigma, local_tgt, pars['n_quad']).item()
		return tgt_idx, log_a


	# Log-probability of the reverse move, from the candidate B back to A:
	# at each chosen site, the current amino acid must be proposed from B's
	# competitors, using the gradient at B. A non-canonical current residue is
	# never a competitor, so that reverse move has probability zero and the
	# forward move is rejected (see _mutate_canonical).
	def _reverse_logprob(self, mu_B: torch.Tensor, cur_idx: list, tgt_idx: list, sigma: float, pars: dict) -> float:
		log_a = 0.
		for site, (cur, tgt) in enumerate(zip(cur_idx, tgt_idx)):
			competitors = self._competition_indices(tgt)
			local_cur = (competitors == cur).nonzero(as_tuple=True)[0]
			if local_cur.numel() == 0:
				return float('-inf')
			log_a += log_pointing_prob(mu_B[site][competitors], sigma, local_cur[0], pars['n_quad']).item()
		return log_a



	# ------------------------------------------------------------------ #
	# Differentiable pass, restricted to the chosen sites: every OTHER   #
	# site is held at its exact discrete (hard one-hot) value; only the  #
	# logits of `sites` are relaxed through softmax(./T_sftm) and        #
	# differentiated. One forward and one backward pass whatever the     #
	# number of sites. Returns one gradient row per site, in `sites`     #
	# order.                                                             #
	# ------------------------------------------------------------------ #
	def _grad_pass_sites(self, eprot: ExtendedProtein, sites: list, pars: dict) -> torch.Tensor:
		hard = eprot.get_probs()  # exact one-hot for the whole sequence, no grad
		site_logits = eprot.logits[sites].detach().clone().requires_grad_(True)
		soft_rows = torch.softmax(site_logits/pars['T_sftm'], dim=-1)
		probs = hard.clone()
		probs[torch.tensor(sites, device=hard.device)] = soft_rows

		am = self.model.predict_attention(sequence_probs=probs)
		U_am = compute_U_am(am, self.ref_eprot.am)
		entropy = compute_entropy(soft_rows, pars['eps'])
		U = pars['lambda_am']*U_am + pars['lambda_S']*entropy

		lambda_structure_ce = pars.get('lambda_structure_ce', 0.)
		if lambda_structure_ce > 0.:
			structure_logits = self.model.predict_structure_logits(sequence_probs=probs)
			U_structure_ce = compute_U_structure_ce(
				structure_logits[1:-1].unsqueeze(0),
				self.ref_structure_tokens.unsqueeze(0),
			)
			U = U + lambda_structure_ce*U_structure_ce

		U.backward()
		grad = site_logits.grad.detach().clone()
		return grad

	# Single-site form, as used by the gradient-method comparison tests.
	def _grad_pass_site(self, eprot: ExtendedProtein, site: int, pars: dict) -> torch.Tensor:
		return self._grad_pass_sites(eprot, [site], pars)[0]

	# ------------------------------------------------------------------ #
	# Exact pass: U evaluated on the true discrete sequence (no softmax  #
	# relaxation, no entropy term), used for the Metropolis ratio.       #
	# Return signature is unchanged (U, U_am, eprot) for backward        #
	# compatibility with every existing caller; U_structure_ce, when in  #
	# use, is stashed on eprot itself (eprot.U_structure_ce, a formally  #
	# declared ExtendedProtein field -- it's an attrs slotted class, so  #
	# a plain dynamic eprot.foo = ... assignment would AttributeError),  #
	# same pattern as eprot.am below. Always a CPU tensor (torch.tensor  #
	# (0.) when lambda_structure_ce is unused) so eprot.save()'s         #
	# unconditional .to("cpu") over every field keeps working; call      #
	# .item() at read sites that want a plain float.                    #
	# ------------------------------------------------------------------ #
	def _exact_energy(self, eprot: ExtendedProtein, pars: dict):
		with torch.no_grad():
			am = self.model.predict_attention(sequence_probs=eprot.get_probs())
			U_am = compute_U_am(am, self.ref_eprot.am)
			U = pars['lambda_am']*U_am

			lambda_structure_ce = pars.get('lambda_structure_ce', 0.)
			if lambda_structure_ce > 0.:
				structure_logits = self.model.predict_structure_logits(sequence_probs=eprot.get_probs())
				U_structure_ce = compute_U_structure_ce(
					structure_logits[1:-1].unsqueeze(0),
					self.ref_structure_tokens.unsqueeze(0),
				)
				U = U + lambda_structure_ce*U_structure_ce
				eprot.U_structure_ce = U_structure_ce.detach().clone().to("cpu")
			else:
				eprot.U_structure_ce = torch.tensor(0.)
		eprot.am = am.detach().clone().to("cpu")
		return U.item(), U_am.item(), eprot

	def _substitute(self, eprot: ExtendedProtein, sites: list, new_idxs: list) -> ExtendedProtein:
		vocab = C.SEQUENCE_USED_VOCAB
		seq = list(eprot.sequence)
		for site, new_idx in zip(sites, new_idxs):
			seq[site] = vocab[new_idx]
		eprot_new = ExtendedProtein(sequence="".join(seq), requires_grad=True, device=eprot.device)
		eprot_new.expand()
		return eprot_new

	# Starting mutations, drawn from the 20 canonical residues. The old start
	# used utils.operations.mutate, which draws from all of SEQUENCE_USED_VOCAB,
	# including X, B, U, Z and O. The sampler never proposes those, and a site
	# that holds one can never change (see _reverse_logprob), so about one start
	# in five was frozen for the whole run (DEVLOG 2026-10-15). Used only when
	# init_muts > 0; init_muts=0 starts at the reference and draws nothing.
	def _mutate_canonical(self, ref_seq: str, mutations: int) -> str:
		canonical = [c for c in C.SEQUENCE_USED_VOCAB if c not in _NONCANONICAL_RESIDUES]
		sites = torch.randint(0, len(ref_seq), (mutations,), generator=self.generator.get(), device=self.generator.device).tolist()
		idxs = torch.randint(0, len(canonical), (mutations,), generator=self.generator.get(), device=self.generator.device).tolist()
		seq = list(ref_seq)
		for site, idx in zip(sites, idxs):
			seq[site] = canonical[idx]
		return "".join(seq)

	def _competition_indices(self, exclude_idx: int) -> torch.Tensor:
		"""Canonical vocab indices eligible to compete for a given site, minus
		exclude_idx if it's among them (typically the site's own current amino
		acid). Non-canonical residues are never in self._canonical_idx at all,
		so they're never candidates -- not penalized, structurally absent."""
		mask = self._canonical_idx != exclude_idx
		return self._canonical_idx[mask]

	def _extract_momenta(self, shape, T, M):
		p = torch.randn(*shape, device=self.generator.device, generator=self.generator.get()) * math.sqrt(T*M)
		p -= p.mean(axis=-1, keepdim=True)
		return p



	def _setup(
			self,
			ref_seq: str,
			pars: dict,
			settings: dict | None = None,
	):
		# 1. PARS
		assert is_subset(pars.keys(), self.defpars.keys()), f"{self.name}._setup(): unexpected key in inputted pars dictionary. Expected keys are: {list(self.defpars.keys())}."
		for key, (value, typ) in self.defpars.items():
			if value is None:
				assert key in pars.keys(), f"{self.name}._setup(): necessary key '{key}' missing from the inputted pars dictionary."
			else:
				if key not in pars:
					pars[key] = value
				else:
					try:
						pars[key] = typ(pars[key])
					except ValueError:
						raise ValueError(f"{self.name}.setup(): pars '{key}' type should be {typ}, but found {type(pars[key])}.")

		assert all([v>=0. for k,v in pars.items() if k in ["lambda_am", "lambda_structure_ce", "lambda_S", "init_muts"]]), (
			f'{self.name}._setup(): invalid value for one of the following keys ("lambda_am", "lambda_structure_ce", "lambda_S", "init_muts"). Allowed values: v>=0.'
		)
		assert (pars["lambda_am"]>0.) or (pars["lambda_structure_ce"]>0.) or (pars["lambda_S"]>0.), (
			f'{self.name}._setup(): invalid value for the keys "lambda_am" ({pars["lambda_am"]}), "lambda_structure_ce" ({pars["lambda_structure_ce"]}) '
			f'and "lambda_S" ({pars["lambda_S"]}). At least one must be positive.'
		)
		assert all([v>0. for k,v in pars.items() if k in ["moves", "T", "dt", "M", "T_sftm", "eps", "n_quad"]]), (
			f'{self.name}._setup(): invalid value for one of the following keys ("moves", "T", "dt", "M", "T_sftm", "eps", "n_quad"). Allowed values: v>0.'
		)
		pars = self._correct_types(pars, "pars")
		assert 1 <= pars["n_sites"] <= len(ref_seq), (
			f'{self.name}._setup(): invalid value for "n_sites" ({pars["n_sites"]}). Allowed values: 1 <= n_sites <= {len(ref_seq)}.'
		)

		# 2. SETTINGS
		if settings is not None:
			assert is_subset(settings.keys(), self.defsettings.keys()), f"{self.name}._setup(): unexpected key in inputted settings dictionary. Expected keys are: {list(self.defsettings.keys())}."
		else:
			settings = {}
		for key, (value, typ) in self.defsettings.items():
			if key not in settings:
				settings[key] = value
				if key in ["results_dir", "weights_dir"]:
					create_path(value)
			else:
				try:
					settings[key] = typ(settings[key])
				except ValueError:
					raise ValueError(f"{self.name}.setup(): settings '{key}' type should be {typ}, but found {type(settings[key])}.")

		assert settings["log_step"] > 0 and settings["print_step"] > 0, (
			f"{self.name}._setup(): 'steps' keys in settings must all be positive, "
			f"but found {settings['log_step']} ('log') and {settings['print_step']} ('print')."
		)

		assert settings["num_threads"] <= 5, f"{self.name}.setup(): invalid value for 'num_threads' variable {settings['num_threads']}. Allowed values: num_threads <= 5."
		torch.set_num_threads(settings["num_threads"])

		settings["device"] = torch.device(settings["device"]) if isinstance(settings["device"], str) else settings["device"]
		if ("cuda" in settings["device"].type) and (not torch.cuda.is_available()):
			settings["device"] = torch.device("cpu")

		self._canonical_idx = self._canonical_idx.to(settings["device"])
		self.model.to(settings["device"])
		self.generator = CustomGenerator(
				seed=pars["seed"],
				device=settings["device"],
		)
		self.ref_eprot = ExtendedProtein(sequence=ref_seq, requires_grad=False, device=self.generator.device)
		self.ref_eprot.expand()
		self.ref_eprot.am = self.model.predict_attention(sequence_probs=self.ref_eprot.get_probs())

		# A non-canonical residue in the reference could never change (see
		# _reverse_logprob), so refuse it rather than sample with frozen sites.
		frozen = sorted(set(ref_seq) & set(_NONCANONICAL_RESIDUES))
		assert not frozen, f"{self.name}._setup(): reference contains non-canonical residues {frozen}, which the sampler can never change."

		if pars['lambda_structure_ce'] > 0.:
			with torch.no_grad():
				ref_structure_logits = self.model.predict_structure_logits(sequence_probs=self.ref_eprot.get_probs())
				self.ref_structure_tokens = ref_structure_logits.argmax(dim=-1)[1:-1]  # drop BOS/EOS

		# 3. CLEAN / RESTART
		if settings["restart"]:
			check_rdir = all([f in os.listdir(settings['results_dir']) for f in ['data.dat', 'pars.txt', 'generator.npy', 'log.pt']])
			check_edir = len([f for f in os.listdir(settings['eprot_dir']) if os.path.isfile(f"{settings['eprot_dir']}/{f}")]) >= 4
			settings["restart"] = check_rdir and check_edir

		if settings["restart"]:
			self.log = torch.load(f'{settings["results_dir"]}/log.pt', weights_only=False)

			data = self.log["data"].copy()
			data = self._correct_types(data, "data")

			self.generator.load(f"{settings['results_dir']}/{self.log['generator']}")
			eprot = ExtendedProtein(
					device=self.generator.device,
					requires_grad=True
			)
			eprot.load(dir=settings['eprot_dir'], suffix=f"_{data['move']}")

		else:
			for d in [settings['results_dir'], settings['eprot_dir']]:
				for fn in os.listdir(d):
					if fn == 'pars.txt': continue
					if os.path.isfile(f'{d}/{fn}'):
						os.remove(f'{d}/{fn}')

			sequence = self._mutate_canonical(self.ref_eprot.sequence, pars['init_muts']) if pars["init_muts"]>0 else self.ref_eprot.sequence
			eprot = ExtendedProtein(sequence=sequence, requires_grad=True, device=self.generator.device)
			eprot.expand()

			data = {
				'move': 0,
				'time': 0.,
				'U': 0.,
				'U_am': 0.,
				'U_structure_ce': 0.,
				'Hd_to_ref': 0,
				'site': 0,
				'cur_aa': '',
				'proposed_aa': '',
				'dU': 0.,
				'log_a_AB': 0.,
				'log_a_BA': 0.,
				'log_ratio': 0.,
				'accepted': 0,
				'acc_moves': 0,
				'acc_rate': 1.,
			}

			U, U_am, eprot = self._exact_energy(eprot, pars)
			data = merge_dict(
				{
					'U': U, 'U_am': U_am, 'U_structure_ce': eprot.U_structure_ce.item(),
					'Hd_to_ref': compute_Hd(eprot.logits, self.ref_eprot.logits),
				},
				data,
			)
			data = self._correct_types(data, "data")
			self._extend_buffer(data, header=True)
			self._save_log(eprot, data, settings)

		self.t0 = ptime() - data["time"]
		self._print_pars(pars, settings)
		self._print_status(data, header=True)

		return (
			eprot,
			data,
			pars,
			settings,
		)



	def _extend_buffer(self, dikt, header=False):
		if header:
			self.buffer.write(
				"# data.dat column definitions:\n"
				"# move        : Monte Carlo move number\n"
				"# time        : CPU time elapsed since simulation start [s]\n"
				"# U           : potential energy of the current accepted sequence\n"
				"# U_am        : attention-map contribution to U\n"
				"# U_structure_ce : structure-token cross-entropy contribution to U (0 if lambda_structure_ce=0)\n"
				"# Hd_to_ref   : Hamming distance between current accepted sequence and reference sequence\n"
				"# site        : sequence position(s) proposed this move (0-indexed), comma-separated when n_sites > 1\n"
				"# cur_aa      : amino acid(s) at the site(s) before this move, comma-separated when n_sites > 1\n"
				"# proposed_aa : amino acid(s) proposed at the site(s) this move, comma-separated when n_sites > 1\n"
				"# dU          : U(proposed) - U(current)\n"
				"# log_a_AB    : log-probability of proposing this move (summed over its sites)\n"
				"# log_a_BA    : log-probability of proposing the reverse move from the candidate (summed over its sites)\n"
				"# log_ratio   : log Metropolis-Hastings ratio, -dU/T + log_a_BA - log_a_AB\n"
				"# accepted    : 1 if proposed move is accepted, 0 if rejected\n"
				"# acc_moves   : cumulative number of accepted Monte Carlo moves\n"
				"# acc_rate    : cumulative Monte Carlo acceptance rate, acc_moves / move\n"
				"#\n"
			)

			header, line = '', ''
			for key in dikt:
				header = header + f'{key}\t'
				line = line + f'{dikt[key]}\t'
			self.buffer.write(f"{header[:-1]}\n{line[:-1]}\n")
		else:
			line = ''
			for key in dikt: line = line + f'{dikt[key]}\t'
			self.buffer.write(f"{line[:-1]}\n")

	def _flush_buffer(self, settings):
		with open(f'{settings["results_dir"]}/data.dat', 'a') as f:
			print(self.buffer.getvalue(), file=f, end="")
		self.buffer.seek(0)
		self.buffer.truncate(0)

	def _save_log(self, eprot, data, settings):
		self._flush_buffer(settings)
		eprot.save(dir=settings["eprot_dir"], suffix=f'_{data["move"]}')
		self.generator.save(f'{settings["results_dir"]}/generator.npy')

		self.log = {
			"data": data.copy(),
			"generator": 'generator.npy',
		}
		torch.save(self.log, f'{settings["results_dir"]}/log.pt')

	def _print_status(self, data, header=False):
		if header:
			print(f'// {self.name} status register:')
			print(f'{self.separator}\n{self.header}\n{self.separator}')

		data['time_h'] = data["time"] / 3600.

		line = ''
		for key, _, fp in self.formatter['sampling']:
			value = data[key]
			value = value if isinstance(value, str) else format(value, f".{fp}f")   # site is a string when n_sites > 1
			line = f'{line}|{value:^12}'

		line = f'{line}|' + ''.join([' ']*5)

		for key, _, fp in self.formatter['efficiency']:
			value = data[key]

			if value is None:
				value = "None"
			else:
				value = format(value, f".{fp}f")

			line = f'{line}|{value:^12}'

		data.pop('time_h')

		print(f'{line}\n{self.separator}')


	def _print_pars(self, pars, settings):
		lines = []
		lines.append(f'# {self.name} parameters summary:')
		lines.append(f'# ')
		lines.append(f'# moves:                      {pars["moves"]:.1e}')
		lines.append(f'# sampling temperature:       {pars["T"]:.1e}')
		lines.append(f'# step size (dt):             {pars["dt"]:.1e}')
		lines.append(f'# logits mass:                {pars["M"]:.2f}')
		lines.append(f'# softmax temperature:        {pars["T_sftm"]:.2f}')
		lines.append(f'# Gauss-Hermite nodes:        {pars["n_quad"]:.0f}')
		lines.append(f'# sites per move:             {pars["n_sites"]}')
		lines.append(f'# attention multiplier:       {pars["lambda_am"]:.1e}')
		lines.append(f'# structure_ce multiplier:    {pars["lambda_structure_ce"]:.1e}')
		lines.append(f'# entropy multiplier:         {pars["lambda_S"]:.1e}')
		lines.append(f'# ')
		lines.append(f'# results directory: {settings["results_dir"]}')
		lines.append(f'# eprot directory:   {settings["eprot_dir"]}')
		lines.append(f'# restart:           {bool(settings["restart"])}')
		lines.append(f'# ')

		max_length = max([len(line) for line in lines])
		print('\n')
		print(''.join(['#'] * (max_length+2)))
		for line in lines:
			line = line + ''.join([' '] * (max_length-len(line)+1)) + '#'
			print(line)
		print(''.join(['#'] * (max_length+2)))
		print()



	def _correct_types(self, d, dname):
		if dname == "pars":
			types_and_keys = [(int, ['moves', 'n_quad', 'init_muts', 'n_sites', 'seed'])]
		else:
			# 'site' is not cast: it holds the comma-separated sites of the move.
			types_and_keys = [
				(int, [
					'move',
					'acc_moves',
					'Hd_to_ref',
				])
			]

		for key in d:
			for _type, keys in types_and_keys:
				if key in keys:
					d[key] = _type(d[key])
		return d

	def _init_attributes(self):
		self.buffer = StringIO()

		self._canonical_idx = torch.tensor(
			[i for i, c in enumerate(C.SEQUENCE_USED_VOCAB) if c not in _NONCANONICAL_RESIDUES],
			dtype=torch.long,
		)

		self.defpars = {
				"moves":(None, int),
				"T": (None, float),
				"dt": (1.0, float),
				"M": (1.0, float),
				"T_sftm": (0.1, float),
				"lambda_am": (1.0, float),
				"lambda_structure_ce": (0.0, float),
				"lambda_S": (0.0, float),
				"n_quad": (40, int),
				"init_muts": (0, int),
				"n_sites": (1, int),
				"eps": (1.0e-9, float),
				"seed": (0, int),
		}

		self.defsettings = {
				"results_dir": ("./results", str),
				"eprot_dir": ("./results/eprot", str),
				"log_step": (1, int),
				"print_step": (1, int),
				"restart": (False, bool),
				"device": ("cpu", str),
				"num_threads": (1, int),
		}

		self.formatter = {
			'sampling': [
				['move',             'move',             0],
				['U',                'U',                5],
				['U_am',             'U_am',             5],
				['U_structure_ce',   'U_ce',             5],
				['Hd_to_ref',		 'Hd_to_ref',        0],
				['site',             'site',             0],
			],
			'efficiency': [
				['dU',               'dU',               5],
				['log_ratio',        'log_ratio',        5],
				['accepted',         'accepted',         0],
				['acc_rate',  		 'acc_rate',	     3],
			],
		}

		self.separator = ''.join(['-']*(13*len(self.formatter['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.formatter['efficiency'])+1))
		self.header = ''
		for _, symbol, _ in self.formatter['sampling']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|' + ''.join([' ']*5)
		for _, symbol, _ in self.formatter['efficiency']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|'
