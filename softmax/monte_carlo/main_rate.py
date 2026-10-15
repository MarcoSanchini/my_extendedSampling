import os
import sys
import argparse

CODE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../code")
)

CUSTOMS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../customs")
)

sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CUSTOMS_DIR)
# references.py (code/tests/) is the single source of truth for the reference
# sequences. Appended, not inserted, so it cannot shadow any module above.
sys.path.append(os.path.join(CODE_DIR, "tests"))

from classes.rate_sampler import ExtendedProteinRateSampler
from utils.general import load_inputs, create_path, find_path
from references import get_reference, header


RESTART_FILES = ["data.dat", "pars.txt", "generator.npy", "log.pt"]


def last_logged_move(data_dat_path):
    with open(data_dat_path) as f:
        rows = [line for line in f if line.strip() and not line.startswith("#")]
    return int(rows[-1].split("\t")[0])


def find_extendable_path(raw_path, pfile):
    """Sim directory to continue with --extend: the one whose pars.txt matches
    pfile in every key except `moves`. Refuses anything the sampler could not
    restart, because its own fallback clears the directory and starts over."""
    wanted = load_inputs(pfile)
    new_moves = wanted.pop("moves")
    matches = []
    for d in sorted(os.listdir(raw_path)):
        path = f"{raw_path}/{d}"
        if not (d.startswith("sim") and os.path.isdir(path)):
            continue
        saved = load_inputs(f"{path}/pars.txt")
        saved.pop("moves", None)
        if saved == wanted:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"--extend: expected one sim directory in {raw_path} with matching "
            f"parameters, found {len(matches)}.")
    path = matches[0]

    missing = [f for f in RESTART_FILES if not os.path.isfile(f"{path}/{f}")]
    eprot_dir = f"{path}/eprot"
    n_eprot = len([f for f in os.listdir(eprot_dir) if os.path.isfile(f"{eprot_dir}/{f}")]) if os.path.isdir(eprot_dir) else 0
    if missing or n_eprot < 4:
        raise FileNotFoundError(
            f"--extend: {path} cannot be restarted (missing {missing}, "
            f"{n_eprot} eprot files). Refusing, since the sampler would clear it.")

    last = last_logged_move(f"{path}/data.dat")
    if new_moves <= last:
        raise ValueError(
            f"--extend: moves={new_moves} is not beyond move {last}, the last "
            f"logged move in {path}.")
    return path


def prepare_directory(args):

    # Load sampler parameters
    pars = {
        "sampler": load_inputs(
            args.pars_file,
            start="## sampler",
            end="##"
        )
    }

    # Load settings
    settings = load_inputs(args.settings_file)

    # Load structure/configuration settings
    config_settings = load_inputs(args.cs_file)

    # Reference sequence, by name from tests/references.py, instead of loading
    # it from a PDB. --ref defaults to protein_g so existing sweeps are unchanged.
    ref_name, ref_seq = get_reference(args.ref)
    pars["protein"] = {
        "ref_name": ref_name,
        "ref_seq": ref_seq
    }

    # Prepare results directory
    create_path(settings["results_dir"])

    if args.extend:
        # Continue the finished run in place: same directory, state restored
        # from its last checkpoint, rows appended to its data.dat.
        settings["results_dir"] = find_extendable_path(settings["results_dir"], args.pars_file)
        settings["restart"] = True
        print(f"Extending existing run in {settings['results_dir']}")
        with open(f"{settings['results_dir']}/extensions.txt", "a") as f:
            print(f"extended to moves={pars['sampler']['moves']} with {args.pars_file}", file=f)
    else:
        settings["results_dir"] = find_path(
            raw_path=settings["results_dir"],
            dname="sim",
            pfile=args.pars_file,
            pname="pars.txt",
            lpfunc=load_inputs
        )

    settings["eprot_dir"] = f"{settings['results_dir']}/eprot"
    create_path(settings["eprot_dir"])

    return pars, settings, config_settings


def main(args):

    print(f"PID: {os.getpid()}\n")

    print("Loading inputs...")
    pars, settings, config_settings = prepare_directory(args)

    print(header(pars['protein']['ref_name'], pars['protein']['ref_seq']))
    print(f"Reference sequence: {pars['protein']['ref_seq']}")
    print(f"Sequence length: {len(pars['protein']['ref_seq'])}")

    print("Initializing rate sampler...")
    sampler = ExtendedProteinRateSampler(
        config_settings=config_settings
    )

    print("Starting the simulation!")

    sampler.sample(
        ref_seq=pars["protein"]["ref_seq"],
        pars=pars["sampler"],
        settings=settings,
    )

    print("\nSimulation completed!")


def create_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pars-file",
        type=str,
        default="rate_inputs/pars.txt",
        help="Path to the parameters file."
    )

    parser.add_argument(
        "--settings-file",
        type=str,
        default="rate_inputs/settings.txt",
        help="Path to the settings file."
    )

    parser.add_argument(
        "--cs-file",
        type=str,
        default="../CONSTANTS.txt",
        help="Path to the structure prediction configuration file."
    )

    parser.add_argument(
        "--ref",
        type=str,
        default="protein_g",
        help="Reference sequence, by name from code/tests/references.py "
             "(protein_g, zero_polymer, zero_polymer_mature, ...). Default "
             "protein_g, the reference every existing sweep was run with."
    )

    parser.add_argument(
        "--extend",
        action="store_true",
        help="Continue the finished run in the matching sim directory to a larger "
             "moves value, restoring its last checkpoint."
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
