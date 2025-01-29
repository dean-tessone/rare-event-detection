import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50)


def cla():
    parser = argparse.ArgumentParser(
        description="list of arguments", formatter_class=formatter
    )

    parser.add_argument(
        "--do_train",
        type=bool,
        required=True,
        help=textwrap.dedent("""Whether to train the model or not"""),
    )
    parser.add_argument(
        "--do_rank",
        type=bool,
        required=True,
        help=textwrap.dedent("""Whether to rank the images or not"""),
    )
    parser.add_argument(
        "--slideID",
        type=str,
        required=True,
        help=textwrap.dedent("""Slide identification number"""),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help=textwrap.dedent("""Standard deviation of the added Gaussian noise"""),
    )
    parser.add_argument(
        "--architec",
        type=str,
        required=True,
        help=textwrap.dedent("""Defines network architecture"""),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        required=False,
        help=textwrap.dedent("""Number of epochs"""),
    )
    parser.add_argument(
        "--zdim",
        type=int,
        default=100,
        required=False,
        help=textwrap.dedent("""Latent dimension"""),
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        required=True,
        help=textwrap.dedent("""Noise type to denoise"""),
    )
    parser.add_argument(
        "--dapi_coef",
        type=float,
        required=True,
        help=textwrap.dedent("""Coefficient of the DAPI channel for ranking"""),
    )
    parser.add_argument(
        "--tritc_coef",
        type=float,
        required=True,
        help=textwrap.dedent("""Coefficient of the TRITC channel for ranking"""),
    )
    parser.add_argument(
        "--cd45_coef",
        type=float,
        required=True,
        help=textwrap.dedent("""Coefficient of the CD45 channel for ranking"""),
    )
    parser.add_argument(
        "--fitc_coef",
        type=float,
        required=True,
        help=textwrap.dedent("""Coefficient of the FITC channel for ranking"""),
    )
    parser.add_argument(
        "--savedir",
        type=str,
        required=True,
        help=textwrap.dedent("""Directory to save model and results"""),
    )

    parser.add_argument(
        "--slide_directory",
        type=str,
        required=True,
        help=textwrap.dedent("""Directory where slides are saved"""),
    )

    return parser.parse_args()
