import numpy as np
import torch

from styletransfer.loss_function import calc_2_moments, calc_style_desc, calc_l2_wass_dist


def test_cal_2_moments_2_dimensions():
    ITERATIONS = 2
    # 2 dimensions
    for _ in range(ITERATIONS):
        x = np.random.rand(20, 3).astype("float32")
        mu = np.mean(x, axis=0, keepdims=True)
        cov = np.cov(x.transpose(), ddof=1)
        mu_t, cov_t = calc_2_moments(torch.from_numpy(x.copy()))
        np.testing.assert_almost_equal(mu_t.numpy(), mu)
        np.testing.assert_almost_equal(cov_t.numpy(), cov)


def test_cal_2_moments_3_dimensions():
    ITERATIONS = 2
    # 3 dimensions
    for _ in range(ITERATIONS):
        x = np.random.rand(20, 20, 3).astype("float32")
        mu_t, cov_t = calc_2_moments(torch.from_numpy(x.copy()))
        x = x.reshape(-1, 3)
        mu = np.mean(x, axis=0, keepdims=True)
        cov = np.cov(x.transpose(), ddof=1)
        np.testing.assert_almost_equal(mu_t.numpy(), mu)
        np.testing.assert_almost_equal(cov_t.numpy(), cov)


def test_cal_style_desc_2_dimensions():
    ITERATIONS = 2
    for _ in range(ITERATIONS):
        x = np.random.rand(20, 3).astype("float32")
        mu_t, tr_cov_t, cov_t = calc_style_desc(
            torch.from_numpy(x.copy()), take_root=False)
        mu_t_2, tr_cov_t_2, root_cov_t = calc_style_desc(
            torch.from_numpy(x.copy()), take_root=True)
        np.testing.assert_almost_equal(mu_t.numpy(), mu_t_2.numpy())
        np.testing.assert_almost_equal(tr_cov_t_2.numpy(), tr_cov_t.numpy())
        np.testing.assert_almost_equal(
            np.mean(x, axis=0, keepdims=True), mu_t.numpy())
        np.testing.assert_almost_equal(
            np.trace(cov_t.numpy()), tr_cov_t.numpy())
        cov = np.cov(x.transpose(), ddof=1)
        np.testing.assert_almost_equal(cov_t.numpy(), cov)
        np.testing.assert_almost_equal(
            np.matmul(root_cov_t.numpy(), root_cov_t.t().numpy()), cov_t.numpy())


def test_cal_style_desc_3_dimensions():
    ITERATIONS = 2
    for _ in range(ITERATIONS):
        x = np.random.rand(20, 20, 3).astype("float32")
        mu_t, tr_cov_t, cov_t = calc_style_desc(
            torch.from_numpy(x.copy()), take_root=False)
        mu_t_2, tr_cov_t_2, root_cov_t = calc_style_desc(
            torch.from_numpy(x.copy()), take_root=True)
        x = x.reshape(-1, 3)
        np.testing.assert_almost_equal(mu_t.numpy(), mu_t_2.numpy())
        np.testing.assert_almost_equal(tr_cov_t_2.numpy(), tr_cov_t.numpy())
        np.testing.assert_almost_equal(
            np.mean(x, axis=0, keepdims=True), mu_t.numpy())
        np.testing.assert_almost_equal(
            np.trace(cov_t.numpy()), tr_cov_t.numpy())
        cov = np.cov(x.transpose(), ddof=1)
        np.testing.assert_almost_equal(cov_t.numpy(), cov)
        np.testing.assert_almost_equal(
            np.matmul(root_cov_t.numpy(), root_cov_t.t().numpy()), cov_t.numpy())


def test_cal_l2_wass_dist():
    ITERATIONS = 5
    for _ in range(ITERATIONS):
        # style
        x = np.random.rand(20, 20, 3).astype("float32")
        # content
        y = np.random.rand(20, 20, 3).astype("float32")
        mu_x, tr_cov_x, root_cov_x = calc_style_desc(
            torch.from_numpy(x.copy()), take_root=True)
        mu_y, tr_cov_y, cov_y = calc_style_desc(
            torch.from_numpy(y.copy()), take_root=False)
        dist = calc_l2_wass_dist(
            [mu_y, tr_cov_y, cov_y],
            [mu_x, tr_cov_x, root_cov_x]
        )
        # calculate in numpy
        mu_diff_squared = np.sum(np.square(mu_x.numpy() - mu_y.numpy()))
        cov_prod = np.matmul(
            np.matmul(root_cov_x.numpy(), cov_y.numpy()),
            root_cov_x.numpy()
        )
        var_overlap = np.sum(
            np.sqrt(np.clip(np.linalg.eigvals(cov_prod), 0.01, 1e10)))
        dist_numpy = mu_diff_squared + tr_cov_x.numpy() + tr_cov_y.numpy() - \
            2 * var_overlap
        np.testing.assert_almost_equal(dist.numpy(), dist_numpy)
