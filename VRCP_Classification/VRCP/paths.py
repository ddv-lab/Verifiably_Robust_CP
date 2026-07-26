"""Single source of truth for the on-disk cache/output directory layout.

Both the experiment driver and Figure_Scripts/*.py use these builders, so
they can't drift out of sync with each other.

Note: the "Robust" path segment is applied to the Adversarial_Examples and
Results directories but not to the Bounds directory. This looks
asymmetric, but changing it would invalidate any Bounds files already
cached under the current layout.
"""

import os


def adversarial_examples_dir(root: str, dataset: str, epsilon, robust: bool, norm_is_inf: bool) -> str:
    d = os.path.join(root, str(dataset), f"epsilon_{epsilon}")
    if robust:
        d = os.path.join(d, "Robust")
    if norm_is_inf:
        d = os.path.join(d, "inf")
    return d


def bounds_dir(root: str, dataset: str, epsilon, norm_is_inf: bool) -> str:
    d = os.path.join(root, str(dataset), f"epsilon_{epsilon}")
    if norm_is_inf:
        d = os.path.join(d, "inf")
    return d


def results_dir(
    root: str,
    dataset: str,
    epsilon,
    sigma_model,
    sigma_smooth,
    n_smooth,
    robust: bool,
    arc: str,
    alpha: float,
    norm_is_inf: bool,
) -> str:
    d = os.path.join(
        root,
        str(dataset),
        f"epsilon_{epsilon}",
        f"sigma_model_{sigma_model}",
        f"sigma_smooth_{sigma_smooth}",
        f"n_smooth_{n_smooth}",
    )
    if robust:
        d = os.path.join(d, "Robust")
    d = os.path.join(d, arc)
    if alpha != 0.1:
        d = os.path.join(d, f"alpha_{alpha}")
    if norm_is_inf:
        d = os.path.join(d, "inf")
    return d
