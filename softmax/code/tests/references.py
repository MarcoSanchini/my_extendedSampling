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

REFERENCES = {
	"protein_g": PROTEIN_G,
	"zero_polymer": ZERO_POLYMER,
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
