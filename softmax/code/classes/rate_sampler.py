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
from utils.energies import compute_U_am, compute_entropy
from utils.operations import compute_Hd, mutate, is_subset, merge_dict
from utils.proposals import log_pointing_prob



# Non-canonical / ambiguous residue codes at the tail of SEQUENCE_USED_VOCAB
# (X=unknown, B=Asx, Z=Glx, U=selenocysteine, O=pyrrolysine). Excluded from
# the proposal competition below: empirically (see tests/test_steepest_descent.py)
# the gradient-informed direction was observed to point at 'X' even when doing
# so *increased* the true energy relative to staying -- these aren't real
# single-residue substitutions and are an unreliable target for a first-order
# proposal. The penalty is a large FINITE value (not -inf) so that a site
# that is already sitting on one of these codes (e.g. via a reference
# sequence or a random init_muts draw) doesn't produce nan in
# log_pointing_prob; its "stay" option is simply also strongly discouraged
# in that (unusual) case, which is an accepted limitation, not a crash.
_NONCANONICAL_RESIDUES = ('X', 'B', 'U', 'Z', 'O')
_NONCANONICAL_PENALTY = -1.0e6



# -------------------------------------------------------------------------- #
# Locally-Balanced Rate ExtendedProtein Sampler                              #
#                                                                             #
# Each move draws one gradient-informed displacement per site (a single      #
# leapfrog half-step, no trajectory/isteps) and reads off, per site, which   #
# amino acid (possibly the current one) the displacement points at. The full #
# joint multi-site candidate is then accepted/rejected with an exact         #
# Metropolis-Hastings correction that uses the true (recomputed) energy of   #
# the candidate and the true reverse-proposal probability, so the target     #
# distribution is exp(-U/T) exactly, not merely to first order.             #
#                                                                             #
# Per move this costs 2 backward passes (grad at A, grad at B) + 1 forward   #
# pass (exact U at B) through ESM3, versus `isteps` (default 100) in the     #
# leapfrog-based ExtendedProteinSampler.                                     #
#                                                                             #
# Note on `dt`: unlike a diffusive/Langevin step size, here dt should be     #
# picked in the "informed" regime (not too small). The standardized          #
# gap driving each site's choice scales as ~dt*(grad_j-grad_k)/sqrt(M*T), so #
# dt -> 0 does NOT concentrate the per-site choice on "stay": it makes it    #
# uniform over all K classes (mean and std of the displacement both scale    #
# with dt, but at different powers, so their ratio -> 0). A dt too small     #
# will therefore propose changes at most/all sites every move and be        #
# rejected almost always; dt large enough that the gradient signal          #
# dominates lets already-good sites confidently vote to stay.               #
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
				data["Hd_to_ref"] = compute_Hd(eprot.logits, self.ref_eprot.logits)

			with torch.no_grad():
				data["entropy"] = compute_entropy(eprot.get_probs(pars["T_sftm"]), pars["eps"]).item()

			data["move"] = move
			data["muts_move"] = info["muts_move"]
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
	# One Monte Carlo move: propose a joint multi-site candidate and     #
	# accept/reject it exactly. Returns (eprot, info), where eprot is    #
	# either the accepted candidate or the (unchanged) input.            #
	# ------------------------------------------------------------------ #
	def _step(self, eprot: ExtendedProtein, U_A: float, pars: dict):
		grad_A = self._grad_pass(eprot, pars)

		sigma = pars['dt'] * math.sqrt(pars['T']/pars['M'])
		step_coef = pars['dt']**2. / (2.*pars['M'])
		mu_A = -step_coef*grad_A
		mu_A = self._penalize_noncanonical(mu_A)

		momentum = self._extract_momenta(grad_A.shape, pars['T'], pars['M'])
		delta_x = pars['dt']*momentum/pars['M'] - step_coef*grad_A
		delta_x = delta_x - delta_x.mean(dim=-1, keepdim=True)
		delta_x = self._penalize_noncanonical(delta_x)

		cur_idx = eprot.logits.argmax(dim=-1)
		tgt_idx = delta_x.argmax(dim=-1)
		muts_move = int((tgt_idx != cur_idx).sum().item())

		info = {
			"muts_move": muts_move,
			"U_B": U_A, "U_am_B": None,
			"dU": 0., "log_a_AB": 0., "log_a_BA": 0., "log_ratio": 0.,
			"accepted": None,
		}

		# Nothing proposed: every site's own amino acid already wins its
		# competition. This is a legitimate outcome (see dt note above),
		# not a bug -- it's an identity move, no Hastings correction needed.
		if muts_move == 0:
			return eprot, info

		log_a_AB = log_pointing_prob(mu_A, sigma, tgt_idx, pars['n_quad']).sum().item()

		eprot_B = self._collapse_to(eprot, tgt_idx)
		grad_B = self._grad_pass(eprot_B, pars)
		mu_B = -step_coef*grad_B
		mu_B = self._penalize_noncanonical(mu_B)
		log_a_BA = log_pointing_prob(mu_B, sigma, cur_idx, pars['n_quad']).sum().item()

		U_B, U_am_B, eprot_B = self._exact_energy(eprot_B, pars)

		dU = U_B - U_A
		log_ratio = -dU/pars['T'] + (log_a_BA - log_a_AB)

		u = torch.rand(1, device=self.generator.device, generator=self.generator.get()).item()
		accepted = math.log(max(u, 1e-300)) <= min(0., log_ratio)

		info.update({
			"U_B": U_B, "U_am_B": U_am_B,
			"dU": dU, "log_a_AB": log_a_AB, "log_a_BA": log_a_BA, "log_ratio": log_ratio,
			"accepted": int(accepted),
		})

		return (eprot_B, info) if accepted else (eprot, info)



	# ------------------------------------------------------------------ #
	# Differentiable pass: U evaluated on the softmax-relaxed (T_sftm)   #
	# probabilities, used only to extract grad w.r.t. eprot.logits.      #
	# ------------------------------------------------------------------ #
	def _grad_pass(self, eprot: ExtendedProtein, pars: dict) -> torch.Tensor:
		probs = eprot.get_probs(pars['T_sftm'])
		am = self.model.predict_attention(sequence_probs=probs)
		U_am = compute_U_am(am, self.ref_eprot.am)
		entropy = compute_entropy(probs, pars['eps'])
		U = pars['lambda_am']*U_am + pars['lambda_S']*entropy

		eprot.logits.grad = None
		U.backward()
		grad = eprot.logits.grad.detach().clone()
		eprot.logits.grad = None
		return grad

	# ------------------------------------------------------------------ #
	# Exact pass: U evaluated on the true discrete sequence (no softmax  #
	# relaxation, no entropy term), used for the Metropolis ratio.       #
	# ------------------------------------------------------------------ #
	def _exact_energy(self, eprot: ExtendedProtein, pars: dict):
		with torch.no_grad():
			am = self.model.predict_attention(sequence_probs=eprot.get_probs())
			U_am = compute_U_am(am, self.ref_eprot.am)
			U = pars['lambda_am']*U_am
		eprot.am = am.detach().clone().to("cpu")
		return U.item(), U_am.item(), eprot

	def _collapse_to(self, eprot: ExtendedProtein, tgt_idx: torch.Tensor) -> ExtendedProtein:
		vocab = C.SEQUENCE_USED_VOCAB
		new_seq = "".join(vocab[t] for t in tgt_idx.tolist())
		eprot_B = ExtendedProtein(sequence=new_seq, requires_grad=True, device=eprot.device)
		eprot_B.expand()
		return eprot_B

	def _penalize_noncanonical(self, x: torch.Tensor) -> torch.Tensor:
		if self._noncanonical_idx.numel() == 0:
			return x
		idx = self._noncanonical_idx.to(x.device)
		x = x.clone()
		x[..., idx] = x[..., idx] + _NONCANONICAL_PENALTY
		return x

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

		assert all([v>=0. for k,v in pars.items() if k in ["lambda_am", "lambda_S", "init_muts"]]), (
			f'{self.name}._setup(): invalid value for one of the following keys ("lambda_am", "lambda_S", "init_muts"). Allowed values: v>=0.'
		)
		assert (pars["lambda_am"]>0.) or (pars["lambda_S"]>0.), (
			f'{self.name}._setup(): invalid value for the keys "lambda_am" ({pars["lambda_am"]}) and "lambda_S" ({pars["lambda_S"]}). At least one must be positive.'
		)
		assert all([v>0. for k,v in pars.items() if k in ["moves", "T", "dt", "M", "T_sftm", "eps", "n_quad"]]), (
			f'{self.name}._setup(): invalid value for one of the following keys ("moves", "T", "dt", "M", "T_sftm", "eps", "n_quad"). Allowed values: v>0.'
		)
		pars = self._correct_types(pars, "pars")

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

		self.model.to(settings["device"])
		self.generator = CustomGenerator(
				seed=pars["seed"],
				device=settings["device"],
		)
		self.ref_eprot = ExtendedProtein(sequence=ref_seq, requires_grad=False, device=self.generator.device)
		self.ref_eprot.expand()
		self.ref_eprot.am = self.model.predict_attention(sequence_probs=self.ref_eprot.get_probs())

		# 3. CLEAN / RESTART
		if settings["restart"]:
			check_rdir = all([f in os.listdir(settings['results_dir']) for f in ['data.dat', 'pars.txt', 'generator.npy', 'log.pt']])
			check_edir = len([f for f in os.listdir(settings['eprot_dir']) if os.path.isfile(f"{settings['eprot_dir']}/{f}")]) >= 4
			settings["restart"] = check_rdir and check_edir

		if settings["restart"]:
			self.log = torch.load(f'{settings["results_dir"]}/log.pt')

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

			sequence = mutate(self.ref_eprot.sequence, pars['init_muts'], self.generator.get()) if pars["init_muts"]>0 else self.ref_eprot.sequence
			eprot = ExtendedProtein(sequence=sequence, requires_grad=True, device=self.generator.device)
			eprot.expand()

			data = {
				'move': 0,
				'time': 0.,
				'U': 0.,
				'U_am': 0.,
				'entropy': 0.,
				'Hd_to_ref': 0,
				'muts_move': 0,
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
					'U': U, 'U_am': U_am,
					'entropy': compute_entropy(eprot.get_probs(pars["T_sftm"]), pars["eps"]).item(),
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
				"# entropy     : Shannon entropy of the softmax probabilities\n"
				"# Hd_to_ref   : Hamming distance between current accepted sequence and reference sequence\n"
				"# muts_move   : number of sites proposed to change in this move (0 = identity move, no Hastings step run)\n"
				"# dU          : U(proposed) - U(current), only meaningful when muts_move>0\n"
				"# log_a_AB    : log-probability of proposing B from A (sum over sites)\n"
				"# log_a_BA    : log-probability of proposing A back from B (sum over sites)\n"
				"# log_ratio   : log Metropolis-Hastings ratio, -dU/T + log_a_BA - log_a_AB\n"
				"# accepted    : 1 if proposed move is accepted, 0 if rejected, None if muts_move==0\n"
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
			line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'

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
		lines.append(f'# attention multiplier:       {pars["lambda_am"]:.1e}')
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
			types_and_keys = [(int, ['moves', 'n_quad', 'init_muts', 'seed'])]
		else:
			types_and_keys = [
				(int, [
					'move',
					'acc_moves',
					'Hd_to_ref',
					'muts_move',
				])
			]

		for key in d:
			for _type, keys in types_and_keys:
				if key in keys:
					d[key] = _type(d[key])
		return d

	def _init_attributes(self):
		self.buffer = StringIO()

		self._noncanonical_idx = torch.tensor(
			[i for i, c in enumerate(C.SEQUENCE_USED_VOCAB) if c in _NONCANONICAL_RESIDUES],
			dtype=torch.long,
		)

		self.defpars = {
				"moves":(None, int),
				"T": (None, float),
				"dt": (1.0, float),
				"M": (1.0, float),
				"T_sftm": (0.1, float),
				"lambda_am": (1.0, float),
				"lambda_S": (0.0, float),
				"n_quad": (40, int),
				"init_muts": (0, int),
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
				['entropy',          'entropy',          5],
				['Hd_to_ref',		 'Hd_to_ref',        0],
				['muts_move',        'muts_move',        0],
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
