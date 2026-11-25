"""Command-line interface."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="worldperturb", description="WorldPerturb tools")
    parser.add_argument("--version", action="version", version="worldperturb 0.1.0")
    parser.parse_args()


if __name__ == "__main__":
    main()
