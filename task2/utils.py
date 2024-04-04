import argparse


def parse_args():
    """
    Parse arguments as per requirements.
    """
    parser = argparse.ArgumentParser(description="Train ViT for COMP0197 group task 2.")

    parser.add_argument(
        "--sampling_method",
        type=int,
        default=1,
        choices=[1, 2],
        help="Sampling method: 1 for beta distribution, 2 for uniform distribution.",
    )

    args = parser.parse_args()

    return args
