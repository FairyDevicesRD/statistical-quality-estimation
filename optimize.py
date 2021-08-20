import argparse
import csv
import collections
import logging
import math
import random
import traceback
import pathlib
import pickle
import warnings

import dirichlet
import joblib
import numpy as np
import scipy.stats
import scipy.optimize
import sklearn
from sklearn.linear_model import LogisticRegression
import toml

logging.basicConfig(
    format="%(asctime)s %(processName)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO)
logger = logging.getLogger()

r_epsilon = 0.01
q_epsilon = 0.01
epsilon = 1e-5


def q2alpha(q, cl2a_list, alpha_init, config):
    n_class = config["data"]["n_class"]
    tol = config["optimize"]["dirichlet"]["tol"]
    max_iter = config["optimize"]["dirichlet"]["max_iter"]
    prior_std = config["model"]["alpha_prior_std"]

    alpha = alpha_init.copy()
    for (creator_id, cls) in cl2a_list:
        artifact_ids = cl2a_list[(creator_id, cls)]
        n_artifact = len(artifact_ids)
        d = np.empty((n_artifact, n_class))
        for i, aid in enumerate(artifact_ids):
            d[i, :] = q[aid]

        try:
            a = dirichlet.mle(
                d, tol=tol, maxiter=max_iter, prior_std=prior_std)
        except dirichlet.dirichlet.NotConvergingError as e:
            logger.error(e)
            logger.error("keep the current value a[{}] = {}"
                         .format((creator_id, cls), alpha[(creator_id, cls)]))
        else:
            alpha[(creator_id, cls)] = a
            logger.debug("a[{}] = {}".format((creator_id, cls), a))

    return alpha


def q2beta(q, r2al_list, b_init, config):
    n_class = config["data"]["n_class"]
    use_atanh = config["model"]["use_atanh"]
    prior_std = config["model"]["beta_prior_std"]

    beta = b_init.copy()
    for reviewer_id in r2al_list:
        artifact_ids = [row[0] for row in r2al_list[reviewer_id]]
        ys = [row[1] for row in r2al_list[reviewer_id]]

        n_artifact = len(artifact_ids)
        d = np.empty((n_artifact, n_class))
        for i, aid in enumerate(artifact_ids):
            d[i, :] = q[aid]

        if use_atanh:
            d = np.arctanh(d)

        clf = LogisticRegression(
            random_state=0,
            solver="lbfgs",
            multi_class="multinomial",
            fit_intercept=False,
            warm_start=True,
            C=2.0 * prior_std ** 2,
            # n_jobs=None,
        )
        clf.coef_ = b_init[reviewer_id]
        clf.intercept_ = 0
        try:
            m = clf.fit(d, ys)
        except Exception as e:
            logger.error(e)
            logger.error("keep the current value b[{}] = {}"
                         .format(reviewer_id, beta[reviewer_id]))
            continue
        else:
            b = m.coef_
            if np.isnan(b).any() or np.isinf(b).any():
                logger.error("invalid value; keep the current value b[{}] = {}"
                             .format(reviewer_id, beta[reviewer_id]))
            else:
                beta[reviewer_id] = b
                logger.debug("b[{}] = {}".format(reviewer_id, b))
    return beta


def make_constraint(n_class):
    """
    # when n_class = 4
    c = scipy.optimize.LinearConstraint(
        A=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        lb=[epsilon, epsilon, epsilon, epsilon, 1.0],
        ub=[1.0 - epsilon, 1.0 - epsilon, 1.0 - epsilon, 1 - epsilon, 1.0],
        keep_feasible=True,
    )
    """
    m = np.identity(n_class)
    m = np.vstack([m, np.ones(n_class)])

    lb = [epsilon] * n_class
    lb.append(1.0)
    ub = [1.0 - epsilon] * n_class
    ub.append(1.0)

    c = scipy.optimize.LinearConstraint(
        A=m,
        lb=lb,
        ub=ub,
        keep_feasible=True,
    )
    return c


def alpha_beta2q(alpha, beta, qs_init, a2cl, a2rl_list, LC, config):
    qs_optimized = qs_init.copy()

    tasks = []
    for artifact_id in a2cl:
        a = alpha[a2cl[artifact_id]]
        bs = []
        ys = []
        for (reviewer_id, cls_reviewer) in a2rl_list[artifact_id]:
            bs.append(beta[reviewer_id])
            ys.append(cls_reviewer)  # should be 0-based index

        t = joblib.delayed(run_minimize)(
            artifact_id,
            log_penalty,
            qs_init[artifact_id],
            args=(a, bs, ys, config),
            method="trust-constr",
            jac="2-point",
            hess=scipy.optimize.BFGS(),
            constraints=LC,
            options={
                # "maxiter": 100,
                # "initial_tr_radius": 1e-1,
                "initial_constr_penalty": 1e4,
                # "verbose": 3,
            },
        )
        tasks.append(t)

    results = joblib.Parallel(n_jobs=-1, verbose=2)(tasks)
    for artifact_id, opt in results:
        if isinstance(opt, Exception):
            logger.error(opt)
            logger.error("error happened during the optimizing of q,"
                         " keep the current value; q[{}] = {}"
                         .format(artifact_id, qs_init[artifact_id]))
            continue

        if not opt.success:
            logger.debug("imcomplete optimization")

        q_opt = opt.x
        if np.isnan(q_opt).any() or np.isinf(q_opt).any():
            logger.error("invalid value; keep the current q")
            continue

        qs_optimized[artifact_id] = q_opt
        logger.debug("q[{}] = {}".format(artifact_id, q_opt))

    return qs_optimized


def run_minimize(artifact_id, *args, **kwargs):
    warnings.filterwarnings(
        "ignore", category=UserWarning,
        message="delta_grad == 0.0.", module="scipy.optimize")
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning,
        message="divide by zero encountered in log",
        module="sklearn.linear_model")
    warnings.filterwarnings(
        "ignore", category=sklearn.exceptions.ConvergenceWarning,
        message="lbfgs failed to converge (status=1)", module="sklearn")

    try:
        return artifact_id, scipy.optimize.minimize(*args, **kwargs)
    except ValueError as e:
        logger.error(traceback.format_exc())
        return artifact_id, e


def log_penalty(q, a, bs, ys, config):
    use_atanh = config["model"]["use_atanh"]
    weight_review = config["model"]["weight_review"]

    logger.debug("q = {}, sum(q) = {}".format(q, np.sum(q)))

    is_valid = (q >= epsilon).all() and (q <= 1.0 - epsilon).all()
    if not is_valid:
        # reject invalid candidate
        return np.finfo("float32").max

    q = q[np.newaxis, :]
    ll = 0.0
    ll += dirichlet.loglikelihood(q, a)
    logger.debug("l_a = {}".format(dirichlet.loglikelihood(q, a)))

    for b, y in zip(bs, ys):
        clf = LogisticRegression(
            multi_class="multinomial",
            fit_intercept=False,
        )
        clf.coef_ = b
        clf.intercept_ = 0.0
        if use_atanh:
            p = np.arctanh(q)
        else:
            p = q
        ll_b = clf.predict_log_proba(p)[0][y]
        if ll_b < np.log(np.finfo(ll_b.dtype).tiny):
            logger.debug("ll_b = {} was replaced by {}"
                         .format(ll_b, np.log(np.finfo(ll_b.dtype).tiny)))
            ll_b = np.log(np.finfo(ll_b.dtype).tiny)

        logger.debug("ll_b = {} * {}"
                     .format(weight_review, clf.predict_log_proba(p)[0][y]))

        ll += weight_review * ll_b

    logger.debug("ll = {}".format(ll))
    return -1.0 * ll


def get_mse(q0, q1):
    v = 0.0
    keys0 = frozenset(q0.keys())
    keys1 = frozenset(q1.keys())
    keys = (keys0 & keys1)
    n = len(keys)
    for k in keys:
        v += np.linalg.norm(q0[k] - q1[k], ord=2)
    mse = v / n
    return mse


def get_loglikelihood(q, alpha, beta, data, config):
    alpha_prior_std = config["model"]["alpha_prior_std"]
    beta_prior_std = config["model"]["alpha_prior_std"]
    n_class = config["data"]["n_class"]
    use_atanh = config["model"]["use_atanh"]
    weight_review = config["model"]["weight_review"]

    norm_alpha = scipy.stats.norm(0.0, alpha_prior_std)
    norm_beta = scipy.stats.norm(0.0, beta_prior_std)

    ll = 0.0
    dof = 0
    # prior
    for (creator_id, cls) in alpha:
        a = alpha[(creator_id, cls)]
        ll_a_prior = norm_alpha.logpdf(a.sum())
        logger.debug("ll_a_prior[{}] = {}"
                     .format((creator_id, cls), ll_a_prior))
        ll += ll_a_prior
        dof += np.sum(a.shape)

    for reviewer_id in beta:
        b = beta[reviewer_id]
        ll_b_prior = norm_beta.logpdf(b.flatten()).sum()
        logger.debug("ll_b_prior[{}] = {}".format(reviewer_id, ll_b_prior))
        ll += ll_b_prior
        dof += np.sum(b.shape)

    # creation
    cl2a_list = data["creator_cls_to_artifact_list"]
    for (creator_id, cls) in alpha:
        a = alpha[(creator_id, cls)]
        artifact_ids = cl2a_list[(creator_id, cls)]
        n_artifact = len(artifact_ids)
        x = np.empty(shape=(n_artifact, n_class))

        for idx, aid in enumerate(artifact_ids):
            x[idx, :] = q[aid]

        ll_generation = dirichlet.loglikelihood(x, a)
        logger.debug("ll_creation[{}] = {}"
                     .format((creator_id, cls), ll_generation))
        ll += ll_generation
        dof += n_artifact * n_class

    # review
    r2ac_list = data["reviewer_to_artifact_cls_list"]
    for reviewer_id in beta:
        b = beta[reviewer_id]

        artifact_ids = []
        ys = []
        for aid, y in r2ac_list[reviewer_id]:
            artifact_ids.append(aid)
            ys.append(y)

        n_artifact = len(artifact_ids)
        x = np.empty((n_artifact, n_class))
        for idx, aid in enumerate(artifact_ids):
            x[idx, :] = q[aid]

        if use_atanh:
            x = np.arctanh(x)

        clf = LogisticRegression(
            multi_class='multinomial',
            fit_intercept=False,
        )
        clf.coef_ = b
        clf.intercept_ = 0.0
        log_probs = clf.predict_log_proba(x)

        ll_review = 0.0
        for logp, y in zip(log_probs, ys):
            if logp[y] < np.log(np.finfo(logp.dtype).tiny):
                logger.warning(
                    "logp = {} was replaced by {}"
                    .format(logp[y], np.log(np.finfo(logp.dtype).tiny)))
                ll_review += np.log(np.finfo(logp.dtype).tiny)
            else:
                ll_review += logp[y]
        logger.debug("ll_review[{}] = {}".format(reviewer_id, ll_review))
        ll += weight_review * ll_review
        dof += n_artifact * n_class

    return ll / dof


def save(i, q, alpha, beta, config):
    exp_dir = config["data"]["exp_dir"]
    path = exp_dir / "{}.npz".format(i)

    logger.info("save the current parameters to {}".format(path))
    np.savez(path, q=q, alpha=alpha, beta=beta)


def load_data(config):
    creation_path = \
        config["data"]["base_dir"] / config["data"]["creation_path"]
    review_path = \
        config["data"]["base_dir"] / config["data"]["review_path"]

    creator_cls_to_artifact_list = collections.defaultdict(list)
    artifact_to_creator_cls = dict()
    with creation_path.open("r") as src:
        reader = csv.reader(src, delimiter="\t")
        header = next(reader)
        for r in reader:
            d = dict(zip(header, r))
            creator_id = int(d["creator_id"])
            cls_creator = int(d["class_id"])  # instruction to creator
            artifact_id = int(d["artifact_id"])

            creator_cls_to_artifact_list[(creator_id, cls_creator)]\
                .append(artifact_id)
            artifact_to_creator_cls[artifact_id] = (creator_id, cls_creator)

    reviewer_to_artifact_cls_list = collections.defaultdict(list)
    artifact_to_reviewer_cls_list = collections.defaultdict(list)
    with review_path.open("r") as src:
        reader = csv.reader(src, delimiter="\t")
        header = next(reader)
        for r in reader:
            d = dict(zip(header, r))
            reviewer_id = int(d["reviewer_id"])
            artifact_id = int(d["artifact_id"])
            cls_reviewer = int(d["class_id"])  # answer by reviewer

            reviewer_to_artifact_cls_list[reviewer_id]\
                .append((artifact_id, cls_reviewer))
            artifact_to_reviewer_cls_list[artifact_id]\
                .append((reviewer_id, cls_reviewer))

    reviewer_ids = set(reviewer_to_artifact_cls_list)
    qualification_accuracy = load_qualification_review(config, reviewer_ids)

    data = {
        "creator_cls_to_artifact_list": creator_cls_to_artifact_list,
        "artifact_to_creator_cls": artifact_to_creator_cls,
        "reviewer_to_artifact_cls_list": reviewer_to_artifact_cls_list,
        "artifact_to_reviewer_cls_list": artifact_to_reviewer_cls_list,
        "qualification_accuracy": qualification_accuracy,
    }
    return data


def load_qualification_review(config, reviewer_ids):
    path = config["data"]["qualification_review_path"]
    if not path:
        logger.info("no data of qualification round found")
        accuracy = {}
        acc = 1.0 - r_epsilon
        for rid in reviewer_ids:
            accuracy[rid] = acc
            logger.debug("qualification_accuray[{}] = {}".format(rid, acc))
        return a

    count = {}
    for rid in reviewer_ids:
        count[rid] = [0, 0]

    path = config["data"]["base_dir"] / path
    with path.open("r") as src:
        reader = csv.reader(src, delimiter="\t")
        header = next(reader)
        for r in reader:
            d = dict(zip(header, r))
            reviewer_id = int(d["reviewer_id"])

            count[reviewer_id][0] += 1
            if int(d["class_id_creator"]) == int(d["class_id_reviewer"]):
                count[reviewer_id][1] += 1

    n_total = sum(c[0] for c in count.values())
    m_total = sum(c[1] for c in count.values())
    # avoid r = 0.0, 1.0
    m_total = min(max(m_total, r_epsilon), n_total - r_epsilon)
    acc_average = m_total / n_total

    accuracy = {}
    for reviewer_id in count:
        n, m = count[reviewer_id]
        if n == 0:
            logger.warning(
                "no data of qualification round for reviewer_id = {}; "
                "average accuracy is used"
                .format(reviewer_id))
            accuarcy[reviewer_id] = acc_average
            logger.info("qualification_accuracy[{}] = {}"
                        .format(reviewer_id, acc_average))

        # avoid r = 0.0, 1.0
        m = min(max(m, r_epsilon), n - r_epsilon)
        acc = m / n
        accuracy[reviewer_id] = acc
        logger.debug("qualification_accuray[{}] = {}".format(reviewer_id, acc))

    return accuracy


def initialize(data, config):
    q_init = get_initial_q(data, config)
    alpha_init = get_initial_alpha(q_init, data, config)
    beta_init = get_initial_beta(data, config)
    return (q_init, alpha_init, beta_init)


def get_initial_q(data, config):
    a2cl = data["artifact_to_creator_cls"]
    a2rl_list = data["artifact_to_reviewer_cls_list"]
    q_acc = data["qualification_accuracy"]
    n_class = config["data"]["n_class"]

    q_init = {}
    for artifact_id in a2cl:
        count = collections.defaultdict(float)

        # add q_epsilon to avoid q = 0.0
        # class_id is 0-based
        for class_id in range(n_class):
            count[class_id] += q_epsilon

        creator_id, cls_creator = a2cl[artifact_id]
        count[cls_creator] += 1.0

        for (reviewer_id, cls_reviewer) in a2rl_list[artifact_id]:
            count[cls_reviewer] += q_acc[reviewer_id]

        q = np.array([count[cls] for cls in range(n_class)])
        # random perturbation
        q = q * np.random.uniform(0.99, 1.01, n_class)
        q = q / np.sum(q)
        q_init[artifact_id] = q

        logger.debug("q_init[{}] = {}".format(artifact_id, q))
    return q_init


def get_initial_alpha(q_init, data, config):
    cl2a_list = data["creator_cls_to_artifact_list"]
    n_class = config["data"]["n_class"]

    alpha = {}
    for (creator_id, cls) in cl2a_list:
        artifact_ids = cl2a_list[(creator_id, cls)]
        n_artifact = len(artifact_ids)
        d = np.empty((n_artifact, n_class))
        for i, aid in enumerate(artifact_ids):
            d[i, :] = q_init[aid]
        a = dirichlet.dirichlet._init_a(d)
        alpha[(creator_id, cls)] = a
        logger.debug("a_init[{}] = {}".format((creator_id, cls), a))
    return alpha


def get_initial_beta(data, config):
    q_acc = data["qualification_accuracy"]
    n_class = config["data"]["n_class"]

    beta = {}
    for reviewer_id in q_acc:
        r = q_acc[reviewer_id]
        b = math.log((n_class - 1) * r / (1 - r))
        b = np.diag([b for _ in range(n_class)])
        beta[reviewer_id] = b
        logger.debug("b_init[{}] = {}".format(reviewer_id, b))
    return beta


def optimize(q, alpha, beta, data, config):
    max_iter = config["optimize"]["convergence"]["max_iter"]
    LC = make_constraint(config["data"]["n_class"])

    cl2a_list = data["creator_cls_to_artifact_list"]
    r2al_list = data["reviewer_to_artifact_cls_list"]
    a2cl = data["artifact_to_creator_cls"]
    a2rl_list = data["artifact_to_reviewer_cls_list"]

    i = 0
    ll_prev = None
    while True:
        i += 1
        alpha = q2alpha(q, cl2a_list, alpha, config)
        beta = q2beta(q, r2al_list, beta, config)
        q_prev = q
        q = alpha_beta2q(alpha, beta, q, a2cl, a2rl_list, LC, config)

        save(i, q, alpha, beta, config)

        if i == 1:
            mse = float("nan")
        else:
            mse = get_mse(q_prev, q)

        ll = get_loglikelihood(q, alpha, beta, data, config)
        logger.info("{}-th iteration, mse={}, loglikelihood={}"
                    .format(i, mse, ll))

        if ll_is_converged(ll_prev, ll, config):
            logger.info("loglikelihood is converged")
            break
        if mse_is_converged(mse, config):
            logger.info("mse is converged")
            break
        if i >= max_iter:
            logger.info("exceeded max_iter")
            break

        ll_prev = ll

    logger.info("end")


def ll_is_converged(ll_prev, ll_cur, config):
    if ll_prev is None:
        return False

    atol = config["optimize"]["convergence"]["loglikelihood_atol"]
    rtol = config["optimize"]["convergence"]["loglikelihood_rtol"]
    return np.isclose(
        ll_prev, ll_cur, atol=atol, rtol=rtol)


def mse_is_converged(mse, config):
    if math.isnan(mse):
        return False

    threshold = config["optimize"]["convergence"]["mse_threshold"]
    return mse < threshold


def load_config(config_path):
    logger.info("load config")
    try:
        config = toml.load(config_path)
    except OSError:
        logger.error("cannot read the configuration file: {}"
                     .format(config_path))
        raise
    except toml.TomlDecodeError:
        logger.error("invalid configuration file: {}"
                     .format(config_path))
        raise

    base_dir = config_path.parent / config["data"]["base_dir"]
    exp_dir = base_dir / config["data"]["exp_id"]
    config["config_path"] = config_path.resolve()
    config["data"]["base_dir"] = base_dir.resolve()
    config["data"]["exp_dir"] = exp_dir.resolve()

    logger.info("config: {}".format(config))
    return config


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
        "ignore", category=UserWarning,
        message="delta_grad == 0.0.", module="scipy.optimize")
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning,
        message="divide by zero encountered in log",
        module="sklearn.linear_model")
    warnings.filterwarnings(
        "ignore", category=sklearn.exceptions.ConvergenceWarning,
        message="lbfgs failed to converge (status=1)", module="sklearn")

    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("start")

    config_path = pathlib.Path(args.config_path)
    config = load_config(config_path)
    config["data"]["exp_dir"].mkdir(parents=True, exist_ok=True)

    data = load_data(config)

    q, alpha, beta = initialize(data, config)
    optimize(q, alpha, beta, data, config)

    logger.info("end")
