import argparse
import logging
import pathlib
import re
import warnings

import numpy as np
import dirichlet
from sklearn.linear_model import LogisticRegression

from optimize import load_config, load_data, get_loglikelihood, get_mse

logger = logging.getLogger()


def main(data, src_dir):
    files = []
    for src_path in src_dir.iterdir():
        m = re.search(r"^(\d+?)\.npz$", str(src_path.name))
        if not m:
            continue
        idx = int(m.group(1))
        files.append((idx, src_path))

    files.sort()

    q_prev = None
    for idx, src_path in files:
        logger.info("load {}".format(src_path))
        d = np.load(str(src_path), allow_pickle=True)

        q = d["q"].flat[0]
        alpha = d["alpha"].flat[0]
        beta = d["beta"].flat[0]

        if q_prev:
            mse = get_mse(q_prev, q)
        else:
            mse = float("nan")
        q_prev = q

        ll = get_loglikelihood(q, alpha, beta, data, config)
        logger.info("{}-th iteration, mse={}, loglikelihood={}"
                    .format(idx + 1, mse, ll))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path", dest="config_path",
        default="config.toml",
        help="configuration file in the TOML format")
    parser.add_argument(
        "-v", "--verbose", help="verbose mode",
        action="store_true", default=False)

    args = parser.parse_args(argv)
    logger.info("argments: {}".format(args))
    return args


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning,
        message="divide by zero encountered in log",
        module="sklearn.linear_model")

    logger.info("start")
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    config_path = pathlib.Path(args.config_path)
    config = load_config(config_path)

    data = load_data(config)

    main(data, config["data"]["exp_dir"])

    logger.info("end")
