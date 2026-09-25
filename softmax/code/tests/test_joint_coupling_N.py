"""
Fixed-N coupling sweep -- the follow-up flagged but never run when the
JOINT multi-site design was abandoned for single-site (M=1) moves (see
DEVLOG.txt, 2026-09-19 "pivot to single-site-per-move": "Worth revisiting
once a real N-sweep ... maps out the coupling-vs-N curve"). Built now
specifically to answer a concrete question: is compute_U_am's non-
separability across simultaneously-changed sites (tests/archive/
test_joint_coupling_JOINT_DESIGN.py: sum_isolated=-16.5 but
dU_joint=+401.2 for one 15-site move) already severe at SMALL N, or only
in the "touch almost the whole sequence" regime the old design fell into
by construction (every trial there touched ~51-56 of 58 sites, never
anything smaller)? That was never actually measured.

Unlike the archived script, this does NOT reuse the (now-removed) JOINT
_step() to draw real stochastic multi-site moves -- the production
sampler is single-site only, there is no "get an N-site proposal" API to
call any more. Instead, for a FIXED N chosen by the sweep (not by dt),
this constructs a synthetic N-site move directly: one whole-sequence
relaxed gradient (see whole_sequence_grad below, the same
reimplementation tests/compare_grad_methods.py uses for the same
already-removed method -- see DEVLOG.txt, 2026-09-24 "multi-site
gradients considered..." entry for why this is the correct historical
gradient computation to use here), then for N randomly-chosen sites, each
site's OWN gradient-greedy target (the deterministic j_star each site's
row would pick alone, same construction as test_informedness_vs_dt.py).
Applying every site's individually-best-looking choice simultaneously,
deterministically, isolates the coupling question from proposal-sampling
noise: if coupling is severe even when every site is choosing its own
locally-best option, that's the strongest, least confounded version of
the question to ask before considering any multi-site design.

Same three-way decomposition as the archived script:
  sum_linear   = first-order Taylor prediction (no coupling, no within-
                 site nonlinearity)
  sum_isolated = true single-site effects, applied one at a time against
                 the ORIGINAL sequence (within-site nonlinearity, still no
                 coupling)
  dU_joint     = true effect with all N sites changed at once (the real
                 quantity a joint accept/reject would use)
  nonlin       = sum_isolated - sum_linear
  coupling     = dU_joint - sum_isolated
N=1 is included as a built-in correctness check: with nothing to couple
with, coupling must be exactly (up to floating point) zero and
sum_isolated must equal dU_joint exactly.

Cannot be run without a working ESM3 install + downloaded weights +
network access for the first `from_pretrained` call -- this was NOT
runnable in the sandbox this was written in. Run it from the softmax/code
directory:
    python tests/test_joint_coupling_N.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import statistics
import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from utils.energies import compute_U_am, compute_entropy
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SEED = 0
N_VALUES = [1, 2, 3, 5, 8, 12, 20]
# Raised from 5 to 20 after the first run (test_joint_coupling_N_results.txt,
# see DEVLOG.txt 2026-09-25): the qualitative result (coupling already
# comparable to dU_joint at N=2, growing with no plateau through N=20) was
# unambiguous, but per-N std was large relative to the mean at low N (e.g.
# N=2: mean=-3.60, std=16.67 from 5 trials), so the summary table's exact
# numbers were noisy. 20 trials/N halves the standard error on the mean
# vs. 5 (SE ~ std/sqrt(n)) at ~4x the (still cheap) cost -- see the
# module docstring's cost estimate, which scales linearly in TRIALS_PER_N.
TRIALS_PER_N = 20
CONSISTENCY_TOL = 1.0e-2


def whole_sequence_grad(sampler, eprot, pars):
    """Reimplementation of the earlier (now-removed) whole-sequence-relaxed
    _grad_pass -- see tests/compare_grad_methods.py, which uses the
    identical function for the identical reason. NOT used by the current
    (single-site) production sampler; diagnostic only."""
    probs = eprot.get_probs(pars['T_sftm'])
    am = sampler.model.predict_attention(sequence_probs=probs)
    U_am = compute_U_am(am, sampler.ref_eprot.am)
    entropy = compute_entropy(probs, pars['eps'])
    U = pars['lambda_am']*U_am + pars['lambda_S']*entropy

    eprot.logits.grad = None
    U.backward()
    grad = eprot.logits.grad.detach().clone()
    eprot.logits.grad = None
    return grad


def main():
    pars = {
        "T": 20.0, "dt": 2.0, "M": 1.0, "T_sftm": 0.1,
        "lambda_am": 1.0, "lambda_structure_ce": 0.0, "lambda_S": 0.0,
        "eps": 1.0e-9, "n_quad": 40,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sampler = ExtendedProteinRateSampler(config_settings={})
    sampler.model.to(device)

    from generator.custom_generator import CustomGenerator
    sampler.generator = CustomGenerator(seed=SEED, device=device)

    sampler.ref_eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
    sampler.ref_eprot.expand()
    sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())

    from utils.operations import mutate
    seq_A = mutate(REF_SEQ, 5, sampler.generator.get())
    eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=True, device=device)
    eprot_A.expand()

    print(f"Reference: {REF_SEQ}")
    print(f"Current A: {seq_A}\n")

    U_A_exact, _, _ = sampler._exact_energy(eprot_A.copy(), pars)
    # One whole-sequence relaxed gradient covers every site; seq_A never
    # changes across the sweep, so this is computed once and reused.
    grad_whole = whole_sequence_grad(sampler, eprot_A, pars)

    vocab = C.SEQUENCE_USED_VOCAB
    L = len(seq_A)

    header = f"{'N':>4} {'trial':>6} {'sum_linear':>12} {'sum_isolated':>13} {'dU_joint':>10} {'nonlin':>10} {'coupling':>10}"
    print(header)
    print("-"*len(header))

    per_N_coupling = {N: [] for N in N_VALUES}
    per_N_dU_joint = {N: [] for N in N_VALUES}

    for N in N_VALUES:
        for trial in range(TRIALS_PER_N):
            g = torch.Generator().manual_seed(1000*N + trial)
            sites = torch.randperm(L, generator=g)[:N].tolist()

            targets = {}
            for s in sites:
                cur = int(eprot_A.logits[s].argmax(dim=-1).item())
                competitors = sampler._competition_indices(cur)
                local_best = (-grad_whole[s][competitors]).argmax(dim=-1)
                targets[s] = int(competitors[local_best].item())

            sum_linear, sum_isolated = 0., 0.
            partial_seq = list(seq_A)
            for s in sites:
                cur = vocab.index(seq_A[s])
                tgt = targets[s]

                onehot_diff = torch.zeros(len(vocab), device=device)
                onehot_diff[tgt] += 1.
                onehot_diff[cur] -= 1.
                sum_linear += (grad_whole[s]*onehot_diff).sum().item()

                cand_seq = seq_A[:s] + vocab[tgt] + seq_A[s+1:]
                eprot_c = ExtendedProtein(sequence=cand_seq, requires_grad=False, device=device)
                eprot_c.expand()
                U_c, _, _ = sampler._exact_energy(eprot_c, pars)
                sum_isolated += U_c - U_A_exact

                partial_seq[s] = vocab[tgt]

            eprot_joint = ExtendedProtein(sequence="".join(partial_seq), requires_grad=False, device=device)
            eprot_joint.expand()
            U_joint, _, _ = sampler._exact_energy(eprot_joint, pars)
            dU_joint = U_joint - U_A_exact

            nonlin = sum_isolated - sum_linear
            coupling = dU_joint - sum_isolated

            per_N_coupling[N].append(coupling)
            per_N_dU_joint[N].append(dU_joint)

            print(f"{N:>4} {trial:>6} {sum_linear:>12.4f} {sum_isolated:>13.4f} {dU_joint:>10.4f} {nonlin:>10.4f} {coupling:>10.4f}")

            if N == 1:
                diff = abs(coupling)
                status = "OK" if diff < CONSISTENCY_TOL else "MISMATCH"
                print(f"         [N=1 sanity check: coupling should be ~0, got {coupling:+.5f} -- {status}]")

    print()
    print("nonlin   = sum_isolated - sum_linear  (within-site linearization error)")
    print("coupling = dU_joint - sum_isolated    (0 => U separable across the touched sites; large |value|")
    print("           => the isolated single-site 'best' targets shift once applied together)")

    print()
    print("=== coupling-vs-N summary ===")
    print(f"{'N':>4} {'mean coupling':>14} {'std coupling':>13} {'mean |coupling|':>16} {'mean dU_joint':>14}")
    for N in N_VALUES:
        c = per_N_coupling[N]
        dU = per_N_dU_joint[N]
        mean_c = statistics.mean(c)
        std_c = statistics.stdev(c) if len(c) > 1 else 0.0
        mean_abs_c = statistics.mean(abs(x) for x in c)
        mean_dU = statistics.mean(dU)
        print(f"{N:>4} {mean_c:>14.4f} {std_c:>13.4f} {mean_abs_c:>16.4f} {mean_dU:>14.4f}")
    print()
    print("Read this table looking for WHERE (if anywhere) mean |coupling| starts growing")
    print("comparably to or faster than mean dU_joint itself -- that's the N at which a joint")
    print("accept/reject would start being dominated by cross-site coupling rather than each")
    print("site's own (correctly identified) locally-best choice, i.e. where a constrained")
    print("multi-site design would stop being trustworthy.")


if __name__ == "__main__":
    main()
