"""Fast, CPU-only smoke tests for the VRCP_Regression stack.

Mirrors vrcp_reg/vrcp.py's actual usage pattern (build the CQR quantile
net, wrap it in auto_LiRPA's BoundedModule, compute CROWN bounds, run it
through nonconformist's IcpRegressor) on small synthetic data -- catching
import errors, auto_LiRPA API breaks, and gross numeric regressions from
the torch/auto_LiRPA/numpy dependency bump. No MPE trajectory generation
here (that needs PettingZoo/tianshou and real simulated data) -- that stays
a separate, slower manual integration check.
"""

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from cqr import helper
from nonconformist.cp import IcpRegressor
from nonconformist.nc import BoundedQuantileRegErrFunc, QuantileRegErrFunc, RegressorNc

DEVICE = "cpu"
IN_SHAPE = 4
QUANTILES = [0.1, 0.9]


def _synthetic_data(rng, n):
    x = rng.normal(size=(n, IN_SHAPE)).astype(np.float32)
    y = (x.sum(axis=1) + 0.1 * rng.normal(size=n)).astype(np.float32)
    return x, y


def _build_and_fit_quantile_net(rng):
    estimator = helper.AllQNet_RegressorAdapter(
        model=None, fit_params=None, in_shape=IN_SHAPE, hidden_size=8, quantiles=QUANTILES,
        learn_func=torch.optim.Adam, epochs=3, batch_size=16, dropout=0.1, lr=0.01, wd=1e-6,
        test_ratio=0.2, random_state=0, use_rearrangement=False,
    )
    x_train, y_train = _synthetic_data(rng, 200)
    estimator.fit(x_train, y_train)
    return estimator


def test_quantile_net_fits_and_predicts():
    rng = np.random.default_rng(0)
    estimator = _build_and_fit_quantile_net(rng)
    x_test, _ = _synthetic_data(rng, 10)
    preds = estimator.predict(x_test)
    assert preds.shape == (10, 2)
    assert np.isfinite(preds).all()


def test_crown_bounds_on_quantile_net():
    rng = np.random.default_rng(0)
    estimator = _build_and_fit_quantile_net(rng)

    dummy = torch.zeros(1, IN_SHAPE)
    lirpa_model = BoundedModule(estimator.model, dummy, device=DEVICE)
    lirpa_model.eval()
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.01)

    x_test, _ = _synthetic_data(rng, 5)
    test_input = torch.from_numpy(x_test)
    my_input = BoundedTensor(test_input, ptb)
    with torch.no_grad():
        lb, ub = lirpa_model.compute_bounds(x=(my_input,), method="CROWN")
        clean_out = lirpa_model(test_input)

    assert torch.isfinite(lb).all() and torch.isfinite(ub).all()
    assert (lb <= ub + 1e-4).all()
    assert (lb <= clean_out + 1e-4).all()
    assert (clean_out <= ub + 1e-4).all()


def test_icp_regressor_with_quantile_and_bounded_error_funcs():
    rng = np.random.default_rng(0)
    estimator = _build_and_fit_quantile_net(rng)
    x_cal, y_cal = _synthetic_data(rng, 50)
    x_test, _ = _synthetic_data(rng, 10)
    alpha = 0.1

    nc = RegressorNc(estimator, QuantileRegErrFunc())
    icp = IcpRegressor(nc)
    icp.calibrate(x_cal, y_cal)
    predictions = icp.predict(x_test, significance=alpha)
    assert predictions.shape == (10, 2)
    assert (predictions[:, 0] <= predictions[:, 1] + 1e-6).all()

    # Mirrors vrcp.py's VRCP--C path: fold CROWN bounds (here, fake bounds of
    # the right shape) into the NCF, sized to match the calibration set
    # `.calibrate()` will actually score (x_cal/y_cal), not the test set.
    calib_preds = estimator.predict(x_cal)
    fake_calib_bounds = np.stack([calib_preds[:, 0] - 0.5, calib_preds[:, 1] + 0.5], axis=1)
    bounded_nc = RegressorNc(estimator, BoundedQuantileRegErrFunc(fake_calib_bounds))
    bounded_icp = IcpRegressor(bounded_nc)
    bounded_icp.calibrate(x_cal, y_cal)
    bounded_predictions = bounded_icp.predict(x_test, significance=alpha)
    assert bounded_predictions.shape == (10, 2)
