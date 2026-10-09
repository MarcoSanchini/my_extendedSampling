"""
Fetch experimental PDB entries for the short reference proteins, accepting an
entry ONLY if its own sequence matches the reference.

Written 2026-10-09, and NOT runnable from the cloud session that wrote it: that
environment's network policy refuses files.rcsb.org (HTTP 403 on CONNECT). Run
it on a machine that can reach the PDB. It uses only the standard library.

WHY THE SEQUENCE GATE. The candidate IDs below are starting points, nothing
more. A reference sequence typed from memory was the error class behind the
off-by-16 numbering trap earlier in this investigation, and an entry ID
recalled from memory can be just as wrong. So no ID is trusted: each downloaded
file's SEQRES records are compared with references.py, and an entry is kept only
if its chain is identical to the reference, or contains it, or is contained in
it. Anything else is reported with the differences and discarded.

For each reference the best accepted entry is written to
bridge_test/pdb/<ref>_<ID>.pdb, and bridge_test/pdb/manifest.json records what
was chosen and why: ID, chain, method, resolution, how the sequence relates to
the reference, the URL, the date and a SHA-256 of the file.

    python bridge_test/fetch_pdb.py                  # protein_g, bpti, crambin
    python bridge_test/fetch_pdb.py --refs bpti
    python bridge_test/fetch_pdb.py --ids bpti=5PTI  # try one specific entry
    python bridge_test/fetch_pdb.py --search         # also ask RCSB for entries
                                                     # whose sequence matches

NMR entries are skipped by default (many models, no resolution) and the first
model of anything kept is what the comparison reads.
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

import references as R  # noqa: E402

OUT = HERE / "pdb"

# Starting points only. Each is verified against the reference sequence before
# use; a wrong ID here costs nothing but a rejected download.
CANDIDATES = {
	"protein_g": ["1PGA", "1PGB", "2GB1"],
	"bpti": ["5PTI", "4PTI", "1BPI", "6PTI"],
	"crambin": ["1CRN", "1EJG", "3NIR"],
}

AA3 = {
	"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
	"GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
	"MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
	"TYR": "Y", "VAL": "V", "MSE": "M",
}


def http(url, data=None, timeout=60):
	req = urllib.request.Request(url, data=data, headers={"User-Agent": "bridge_test-fetch_pdb"})
	if data is not None:
		req.add_header("Content-Type", "application/json")
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		return resp.read()


def parse_header(text):
	"""Method, resolution and per-chain SEQRES sequences from a PDB file."""
	method = " ".join(l[10:].strip() for l in text.splitlines() if l.startswith("EXPDTA")).strip()
	res = None
	for line in text.splitlines():
		m = re.match(r"REMARK   2 RESOLUTION\.\s+([\d.]+)", line)
		if m:
			res = float(m.group(1))
			break
	chains = {}
	for line in text.splitlines():
		if line.startswith("SEQRES"):
			parts = line.split()
			chains.setdefault(parts[2], []).extend(AA3.get(r, "X") for r in parts[4:])
	return method, res, {c: "".join(v) for c, v in chains.items()}


def relation(ref, chain):
	"""How a chain's SEQRES relates to the reference, or None if it does not."""
	if chain == ref:
		return "identical"
	if ref in chain:
		return "chain contains reference (extra residues, e.g. a tag)"
	if chain and chain in ref:
		return "chain is a fragment of the reference"
	return None


def hamming(a, b):
	return sum(1 for x, y in zip(a, b) if x != y) if len(a) == len(b) else None


def evaluate(ref_name, ref_seq, pdb_id, allow_nmr):
	"""Download one entry and decide whether it is acceptable."""
	url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
	try:
		raw = http(url)
	except (urllib.error.URLError, OSError) as exc:
		return {"id": pdb_id, "ok": False, "why": f"download failed: {exc}"}
	text = raw.decode("utf-8", errors="replace")
	method, res, chains = parse_header(text)
	if "NMR" in method.upper() and not allow_nmr:
		return {"id": pdb_id, "ok": False, "why": f"NMR entry skipped ({method})"}
	best = None
	for ch, seq in chains.items():
		rel = relation(ref_seq, seq)
		if rel and (best is None or (rel == "identical" and best[2] != "identical")):
			best = (ch, seq, rel)
	if best is None:
		notes = []
		for ch, seq in chains.items():
			h = hamming(ref_seq, seq)
			notes.append(f"chain {ch}: {len(seq)} residues"
						 + (f", {h} differences from the reference" if h is not None else ", different length"))
		return {"id": pdb_id, "ok": False, "why": "sequence does not match: " + "; ".join(notes)}
	ch, seq, rel = best
	n_ca = sum(1 for l in text.splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[21] == ch)
	return {"id": pdb_id, "ok": True, "chain": ch, "method": method, "resolution": res,
			"relation": rel, "url": url, "n_ca": n_ca, "text": text, "raw": raw}


def search_ids(seq):
	"""Entry IDs whose polymer sequence matches, from the RCSB search API."""
	query = {
		"query": {"type": "terminal", "service": "sequence",
				  "parameters": {"evalue_cutoff": 1, "identity_cutoff": 0.99,
								 "sequence_type": "protein", "value": seq}},
		"return_type": "polymer_entity",
		"request_options": {"paginate": {"start": 0, "rows": 15}},
	}
	try:
		data = json.loads(http("https://search.rcsb.org/rcsbsearch/v2/query", json.dumps(query).encode()))
	except (urllib.error.URLError, OSError, ValueError) as exc:
		print(f"    search failed: {exc}")
		return []
	return [r["identifier"].split("_")[0] for r in data.get("result_set", [])]


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--refs", nargs="*", default=list(CANDIDATES), help="which references to fetch")
	ap.add_argument("--ids", nargs="*", default=[], help="extra candidates, as ref=ID")
	ap.add_argument("--search", action="store_true", help="also query RCSB by sequence")
	ap.add_argument("--allow-nmr", action="store_true")
	args = ap.parse_args()

	extra = {}
	for item in args.ids:
		ref, _, pid = item.partition("=")
		extra.setdefault(ref, []).append(pid.upper())

	OUT.mkdir(exist_ok=True)
	manifest_path = OUT / "manifest.json"
	manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

	for ref in args.refs:
		if ref not in R.REFERENCES:
			print(f"unknown reference {ref!r}; available: {sorted(R.REFERENCES)}")
			continue
		seq = R.REFERENCES[ref]
		ids = list(dict.fromkeys(CANDIDATES.get(ref, []) + extra.get(ref, [])))
		print(f"\n=== {ref} (L={len(seq)}) ===")
		if args.search:
			found = search_ids(seq)
			print(f"  RCSB sequence search returned: {', '.join(found) or 'nothing'}")
			ids = list(dict.fromkeys(ids + found))
		results = []
		for pid in ids:
			r = evaluate(ref, seq, pid, args.allow_nmr)
			results.append(r)
			if r["ok"]:
				res = f"{r['resolution']:.2f} A" if r["resolution"] else "no resolution"
				print(f"  {pid}: ACCEPT  chain {r['chain']}, {r['method']}, {res}, {r['n_ca']} CA, {r['relation']}")
			else:
				print(f"  {pid}: reject  {r['why']}")
		good = [r for r in results if r["ok"]]
		if not good:
			print(f"  no acceptable entry for {ref}. Try --search, or --ids {ref}=<ID> for a specific entry.")
			continue
		# identical sequence first, then best (lowest) resolution; missing resolution last
		good.sort(key=lambda r: (r["relation"] != "identical", r["resolution"] if r["resolution"] else 99.0))
		pick = good[0]
		path = OUT / f"{ref}_{pick['id']}.pdb"
		path.write_bytes(pick["raw"])
		manifest[ref] = {
			"id": pick["id"], "chain": pick["chain"], "method": pick["method"],
			"resolution": pick["resolution"], "sequence_relation": pick["relation"],
			"resolved_CA": pick["n_ca"], "url": pick["url"], "file": path.name,
			"sha256": hashlib.sha256(pick["raw"]).hexdigest(),
			"fetched": datetime.date.today().isoformat(),
			"other_accepted": [r["id"] for r in good[1:]],
		}
		print(f"  -> kept {path.name}  (chain {pick['chain']}); other accepted: {manifest[ref]['other_accepted'] or 'none'}")

	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
	print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
	main()
