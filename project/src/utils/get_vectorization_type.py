import argparse


def get_vec(args: argparse.Namespace) -> str:
    if args.vector == "fasttext":
        return "fasttext"
    else:
        return "bow"
