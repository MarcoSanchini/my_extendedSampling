"""
Single source of truth for the reference sequences the diagnostics run on.

Added 2026-10-02 when the whole safety-check suite had to be re-run on
zero_polymer (566 aa) after the joint-coupling results came back with dU an
order of magnitude above protein G's (see DEVLOG.txt). Before this, every
test script carried its own copy of the protein G REF_SEQ literal -- 7
copies in tests/ alone. Re-pointing the suite meant editing each one, and
any file missed (or edited to a subtly different string) would silently
produce results for a DIFFERENT protein than its sibling scripts, with
nothing in the output to reveal it.

Switch reference without editing any file, via the REF environment
variable:
    python tests/test_steepest_descent.py                  # default below
    REF=protein_g python tests/test_steepest_descent.py
    REF=zero_polymer python tests/test_steepest_descent.py
    REF=zero_polymer_mature python tests/test_steepest_descent.py
    REF=polymer_one python tests/test_steepest_descent.py

zero_polymer is a hemagglutinin HA0 precursor, which in the mature protein
is cleaved into two chains (HA1 and HA2) held together by a single
interchain disulfide bridge. The cleaved form and its two chains are set up
in the block below ZERO_POLYMER; read the NUMBERING note there before using
any position from it, because the literature's HA numbering runs 16 behind
the index into ZERO_POLYMER and an off-by-16 there is silent. The
chainbreak-carrying reference, zero_polymer_cleaved, is defined but is NOT
yet runnable -- see CHAINBREAK_UNSUPPORTED.

Every script that uses this prints the ACTIVE NAME and length in its own
header, so a results file always says which protein it is about.

Running the suite on protein_g is the CONTROL, and is worth doing before
trusting any zero_polymer verdict: protein G's numbers are already recorded
(code/protein_g/*_results.txt, and the DEVLOG entries that discuss them),
so if the current code reproduces them, the harness is intact and any
zero_polymer difference is a real property of the longer sequence rather
than a regression. Known protein G landmarks to check against:
  - test_steepest_descent.py: top-1 accuracy 2/10, and site 4 showing a
    gradient-predicted substitution with true dU=+356.4977 (that exact
    outlier has reappeared in every run since 2026-09-21).
  - scan_U_am_landscape.py: compute_U_am(ref_am, ref_am)=0 exactly, and 0
    negative dU across all reference-anchored single mutants.
  - compare_grad_methods.py: cosine similarity >= 0.9995 at every site.
"""
import os


# Protein G, 56 aa. Zambon et al 2024's own target, and the reference behind
# every result in code/protein_g/ and the DEVLOG up to 2026-10-02.
PROTEIN_G = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

# zero_polymer, 566 aa (~10x protein G). One line per 60 residues, so line k
# holds 0-indexed sites 60k..60k+59.
ZERO_POLYMER = (
	"MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQ"
	"SSSTGEICDSPHQILDGKNCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPD"
	"YASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSKNSFFSRLNWLTHLNFKYPAL"
	"NVTMPNNEQFDKLYIWGVHHPGTDKDQIFLYAQASGRITVSTKRSQQTVSPNIGSRPRVR"
	"NIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITP"
	"NGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWE"
	"GMVDGWYGFRHQNSEGRGQAADLKSTQAAIDQINGKLNRLIGKTNEKFHQIEKEFSEVEG"
	"RIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTKKQLRENAEDMGN"
	"GCFKIYHKCDNACIGSIRNGTYDHDVYRDEALNNRFQIKGVELKSGYKDWILWISFAISC"
	"FLLCVALLGFIMWACQKGNIRCNICI"
)

# ---------------------------------------------------------------------------
# zero_polymer, cleaved into two chains
# ---------------------------------------------------------------------------
# zero_polymer is a hemagglutinin HA0 precursor, and HA0 is not one chain in
# the mature protein: it is cut once, into HA1 and HA2, and the two halves
# stay a single molecule only because a disulfide bridge holds them together.
# That is the "more complicated sequence" this block sets up -- same residues
# as zero_polymer, but presented to ESM3 as a two-chain complex.
#
# NUMBERING. The literature numbers HA by the MATURE protein, which starts
# after the 16-residue signal peptide; ZERO_POLYMER above stores the
# precursor, signal peptide included. So mature numbering runs 16 behind the
# string index, and every landmark below is off by 16 if read as a plain
# offset into ZERO_POLYMER:
#
#     mature position n  ==  ZERO_POLYMER[SIGNAL_PEPTIDE_LEN + n - 1]
#
# This is a live trap, not a hypothetical: ZERO_POLYMER[328] (the naive read
# of "Arg329") is T, and ZERO_POLYMER[13] (the naive read of "Cys14") is V.
# Neither raises; both just quietly describe the wrong residue. Use
# mature_to_index() rather than adding 16 by hand.
SIGNAL_PEPTIDE_LEN = 16

# Cleavage site: HA0 is cut immediately after Arg329 (mature numbering),
# ...PEKQT R | G IFGAI..., which is ZERO_POLYMER index 344 -- so chain one is
# everything up to and including that Arg, chain two everything after it.
# The signal peptide is NOT part of either chain: it is removed
# co-translationally, long before HA0 is ever cleaved, so a construct
# carrying both it and the HA1/HA2 cut is not a species that exists. Dropping
# it is also what makes all three landmarks below exact chain-local
# positions. Cost of that choice: 550 residues, not zero_polymer's 566, so
# results here are NOT length-comparable to the recorded zero_polymer runs
# in code/zero_polymer/ -- compare against ZERO_POLYMER_MATURE instead.
CLEAVAGE_SITE_MATURE = 329
_CUT = SIGNAL_PEPTIDE_LEN + CLEAVAGE_SITE_MATURE  # 345; first index of chain two

# Both chains are SLICED from ZERO_POLYMER, never re-typed as literals --
# this module exists because seven hand-copied reference literals could drift
# apart without the output revealing it, and a second copy of 550 residues
# would reintroduce exactly that.
POLYMER_ONE = ZERO_POLYMER[SIGNAL_PEPTIDE_LEN:_CUT]   # HA1, 329 aa, ends ...PEKQTR
POLYMER_TWO = ZERO_POLYMER[_CUT:]                     # HA2, 221 aa, starts GIFGAI...

# The disulfide bridge. Cys14 of polymer_one to Cys137 of polymer_two: the
# interchain bond that keeps the two cleaved halves one molecule. Positions
# are 1-based and CHAIN-LOCAL (polymer_one's own numbering coincides with
# mature HA1 numbering, since the signal peptide is gone).
#
# This is why the two chains must go through ESM3 in ONE forward pass over
# ZERO_POLYMER_CLEAVED, not as two independent predictions: separate passes
# would model two free monomers, which is a different molecule. The bridge
# itself is recorded here, not imposed as a constraint -- whether ESM3
# recovers it unaided is the question worth asking, and pinning the two
# cysteines together would answer it by assumption.
DISULFIDE_BRIDGES = (
	(("polymer_one", 14), ("polymer_two", 137)),
)

CHAIN_BREAK = "|"

# What ESM3 should see: the two chains joined by ESM3's own chainbreak
# character. NOTE the length is 551, one more than the 550 residues, because
# the break occupies a position of its own in the token stream.
ZERO_POLYMER_CLEAVED = POLYMER_ONE + CHAIN_BREAK + POLYMER_TWO

# The same 550 residues with NO chainbreak: one continuous chain, as if the
# cleavage had never happened. This is the control for the cleaved run -- it
# isolates what the chainbreak alone changes, holding residues fixed -- and,
# unlike ZERO_POLYMER_CLEAVED, it runs on the pipeline as it stands today
# (see CHAINBREAK_UNSUPPORTED below).
ZERO_POLYMER_MATURE = POLYMER_ONE + POLYMER_TWO


def mature_to_index(position: int) -> int:
	"""0-based index into ZERO_POLYMER of a 1-based MATURE HA position.
	Use this instead of adding SIGNAL_PEPTIDE_LEN by hand -- see the
	numbering note above for why an off-by-16 here is silent."""
	if not 1 <= position <= len(ZERO_POLYMER) - SIGNAL_PEPTIDE_LEN:
		raise ValueError(
			f"Mature position {position} is outside the mature protein "
			f"(1..{len(ZERO_POLYMER) - SIGNAL_PEPTIDE_LEN})."
		)
	return SIGNAL_PEPTIDE_LEN + position - 1


CHAINS = {
	"polymer_one": POLYMER_ONE,
	"polymer_two": POLYMER_TWO,
}


# ---------------------------------------------------------------------------
# Disulfide calibration controls
# ---------------------------------------------------------------------------
# Added 2026-10-09. protein_g validates that the folding harness works, but it
# has NO cysteines, so it cannot validate the MEASUREMENT -- and the HA2-alone
# run exposed why that matters: a fold at plddt 93.9 with zero clashes put its
# closest cysteine pair at 6.26 A, beyond the 5.67 A geometric ceiling for a
# bonded pair, i.e. no disulfide at all. Two readings were open: ESM3 cannot
# express a disulfide through this backbone+CB decoder, or HA2's disulfides
# genuinely are not formed. These two references decide it, by asking what the
# pipeline does with disulfides whose positions are KNOWN.
#
# Both are small, both carry exactly three disulfides, and between them they
# use all six of their cysteines, so a mistyped sequence cannot satisfy
# _check() below by accident.

# BPTI (bovine pancreatic trypsin inhibitor), UniProt P00974, mature chain,
# 58 aa. Cys at 5, 14, 30, 38, 51, 55.
BPTI = "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"
BPTI_DISULFIDES = ((5, 55), (14, 38), (30, 51))

# Crambin, UniProt P01542, 46 aa. Cys at 3, 4, 16, 26, 32, 40.
# NOTE THE ISOFORM: positions 22 and 25 vary naturally (Pro/Ser, Leu/Ile).
# This is the Pro22/Leu25 variant -- seq[21] == "P" and seq[24] == "L", which
# _check() asserts, so swapping in the other isoform cannot pass silently.
# Any result from this reference should name the variant.
CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN"
CRAMBIN_DISULFIDES = ((3, 40), (4, 32), (16, 26))

# Positions are 1-based and chain-local, same convention as DISULFIDE_BRIDGES.
KNOWN_DISULFIDES = {
	"bpti": BPTI_DISULFIDES,
	"crambin": CRAMBIN_DISULFIDES,
}


def chain_index(chain: str, position: int) -> int:
	"""0-based index into ZERO_POLYMER_CLEAVED of residue `position` (1-based,
	chain-local) of `chain` ("polymer_one" or "polymer_two"). Accounts for the
	chainbreak position, so e.g. the disulfide pair is

	    [chain_index(c, p) for c, p in DISULFIDE_BRIDGES[0]]  -> [13, 466]
	"""
	if chain not in CHAINS:
		raise ValueError(f"Unknown chain {chain!r}. Available: {sorted(CHAINS)}.")
	if not 1 <= position <= len(CHAINS[chain]):
		raise ValueError(
			f"Position {position} is outside {chain} (1..{len(CHAINS[chain])})."
		)
	offset = 0 if chain == "polymer_one" else len(POLYMER_ONE) + len(CHAIN_BREAK)
	return offset + position - 1


REFERENCES = {
	"protein_g": PROTEIN_G,
	"zero_polymer": ZERO_POLYMER,
	# 550 aa, no chainbreak -- the single-chain control for the cleaved run.
	"zero_polymer_mature": ZERO_POLYMER_MATURE,
	# Each chain on its own. Running these is the OTHER control: it is what
	# the model sees when the disulfide is ignored and the halves are treated
	# as free monomers, which is the thing the cleaved reference is meant to
	# differ from.
	"polymer_one": POLYMER_ONE,
	"polymer_two": POLYMER_TWO,
	# Disulfide calibration controls -- see the block above KNOWN_DISULFIDES.
	"bpti": BPTI,
	"crambin": CRAMBIN,
}

# References carrying a chainbreak. Deliberately kept OUT of REFERENCES: the
# pipeline cannot consume them yet, and get_reference() reports that rather
# than handing back a string that fails later and further away. What is
# missing, concretely:
#   - ExtendedProtein.expand() maps each character through
#     C.SEQUENCE_USED_VOCAB, which is the 25 amino acids and has no "|", so
#     "|" raises ValueError there (SEQUENCE_VOCAB has it at 31,
#     SEQUENCE_CHAINBREAK_TOKEN, but USED_VOCAB is what the differentiable
#     logits are built over).
#   - The soft path never reaches token 31 either: encoder
#     format_sequence_probs() pads the 25-wide prob vector into the 64-wide
#     vocab with ZEROS, and index 31 falls in that zero region, so a
#     chainbreak cannot be expressed as sequence_probs as things stand.
#   - ESM3._default() seeds structure tokens with BOS/EOS only; the hard path
#     masked_fills STRUCTURE_CHAINBREAK_TOKEN from the sequence tokens
#     (models/esm3.py), but the custom soft path has no sequence tokens to
#     read, so the structure track would carry a mask where the break is.
#   - utils/operations.py:mutate() samples uniformly over USED_VOCAB and over
#     all L sites, so it would happily overwrite the break with an amino acid
#     and silently re-fuse the chains mid-run.
CHAINBREAK_UNSUPPORTED = {
	"zero_polymer_cleaved": ZERO_POLYMER_CLEAVED,
}

# The reference used when REF is unset. zero_polymer is the current
# investigation; set REF=protein_g for the control run described above.
DEFAULT_REF = "zero_polymer"


def get_reference(name: str | None = None) -> tuple[str, str]:
	"""(name, sequence) for the active reference. `name` overrides the REF
	environment variable, which overrides DEFAULT_REF. Raises on an unknown
	name rather than silently falling back -- a typo'd REF must not quietly
	produce results for the wrong protein, which is the whole point of this
	module."""
	key = name or os.environ.get("REF") or DEFAULT_REF
	key = key.strip()
	if key in CHAINBREAK_UNSUPPORTED:
		# A known reference the pipeline cannot run yet. Say so here, where
		# the reason is available, instead of letting it reach
		# ExtendedProtein.expand() and die on an unexplained
		# "'|' is not in list" several frames away.
		raise NotImplementedError(
			f"Reference {key!r} contains a chainbreak ({CHAIN_BREAK!r}), which the "
			f"pipeline does not handle yet: ExtendedProtein.expand() builds logits "
			f"over C.SEQUENCE_USED_VOCAB (25 amino acids, no chainbreak), the "
			f"encoder's format_sequence_probs() zero-pads over token 31 so the "
			f"break cannot be expressed as sequence_probs, ESM3._default() would "
			f"leave the structure track masked at the break, and "
			f"utils.operations.mutate() would overwrite it. See "
			f"CHAINBREAK_UNSUPPORTED in this module. For a runnable approximation "
			f"use REF=zero_polymer_mature (same 550 residues, single chain)."
		)
	if key not in REFERENCES:
		raise ValueError(
			f"Unknown reference {key!r}. Available: {sorted(REFERENCES)}. "
			f"Set it via the REF environment variable, e.g. REF=protein_g."
		)
	return key, REFERENCES[key]


def header(name: str, seq: str) -> str:
	"""One line identifying the active reference, for a results file's own
	header -- so a saved results file is self-describing."""
	return f"# reference: {name} (L={len(seq)} residues)"


# Import-time invariants. Every landmark above is a derived slice, so a typo
# in a bound, or an edit to the ZERO_POLYMER literal, would shift the chains
# and the disulfide anchors together and produce a plausible-looking wrong
# molecule. These are the cheap checks that make that loud instead.
def _check() -> None:
	assert len(POLYMER_ONE) == CLEAVAGE_SITE_MATURE, (
		f"polymer_one should be {CLEAVAGE_SITE_MATURE} aa (mature HA1), "
		f"got {len(POLYMER_ONE)}"
	)
	# The cleavage site itself: ...QT R | G I ...
	assert POLYMER_ONE[-3:] == "QTR", (
		f"polymer_one should end at Arg329, ...QTR, got ...{POLYMER_ONE[-3:]}"
	)
	assert POLYMER_TWO[:2] == "GI", (
		f"polymer_two should start GI, got {POLYMER_TWO[:2]}"
	)
	# No residues lost or duplicated by the split.
	assert POLYMER_ONE + POLYMER_TWO == ZERO_POLYMER[SIGNAL_PEPTIDE_LEN:]
	assert ZERO_POLYMER_CLEAVED.count(CHAIN_BREAK) == 1
	# Both disulfide anchors must actually be cysteines.
	for pair in DISULFIDE_BRIDGES:
		for chain, position in pair:
			residue = CHAINS[chain][position - 1]
			assert residue == "C", (
				f"disulfide anchor {chain}:{position} should be a cysteine, "
				f"found {residue}"
			)
			# chain_index() must agree with the chain-local lookup.
			assert ZERO_POLYMER_CLEAVED[chain_index(chain, position)] == "C"

	# Disulfide calibration controls. The pairings are the whole point of
	# these references -- a sequence that drifted would calibrate the
	# yardstick against the wrong molecule, which is worse than no control.
	for name, seq, pairs, length in (
		("bpti", BPTI, BPTI_DISULFIDES, 58),
		("crambin", CRAMBIN, CRAMBIN_DISULFIDES, 46),
	):
		assert len(seq) == length, f"{name} should be {length} aa, got {len(seq)}"
		listed = [p for pair in pairs for p in pair]
		assert len(listed) == len(set(listed)), (
			f"{name}: a cysteine appears in two disulfides: {sorted(listed)}"
		)
		for position in listed:
			assert seq[position - 1] == "C", (
				f"{name}:{position} should be a cysteine, found {seq[position - 1]}"
			)
		# Every cysteine must be accounted for, so a mistyped sequence cannot
		# pass by happening to keep the listed positions intact.
		found = {i + 1 for i, a in enumerate(seq) if a == "C"}
		assert found == set(listed), (
			f"{name}: cysteines {sorted(found)} but disulfides cover "
			f"{sorted(listed)}"
		)

	# Crambin isoform guard: this is the Pro22/Leu25 variant, and the other
	# natural isoform (Ser22/Ile25) would otherwise substitute silently.
	assert CRAMBIN[21] == "P" and CRAMBIN[24] == "L", (
		f"CRAMBIN should be the Pro22/Leu25 variant, found "
		f"{CRAMBIN[21]}22/{CRAMBIN[24]}25"
	)


_check()
