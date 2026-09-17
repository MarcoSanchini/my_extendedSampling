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

from classes.rate_sampler import ExtendedProteinRateSampler
from utils.general import load_inputs, create_path, find_path


# Reference protein sequence
REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


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

    # Use the sequence defined above instead of loading it from a PDB
    pars["protein"] = {
        "ref_seq": REF_SEQ
    }

    # Prepare results directory
    create_path(settings["results_dir"])

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

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
