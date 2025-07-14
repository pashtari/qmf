import torch

import qmf


def test_qmf():
    x = torch.randint(0, 256, size=(1, 784, 192))
    qmf = qmf.QMF(rank=5, num_iters=10, verbose=True)
    u, v = qmf.decompose(x)
    return u, v


def test_hosvd_rank_upper_bounds():
    upper_bounds = qmf.hosvd_rank_upper_bounds([100, 5, 6])
    assert tuple(upper_bounds) == (30, 5, 6)


def test_hosvd():
    x = torch.rand(784, 8, 8, 3)
    core, factors = qmf.hosvd(x, rank=(49, 5, 5, 3))
    x_hat = qmf.multi_mode_product(core, factors)
    (x - x_hat).abs().mean()


def test_batched_hosvd():
    x = torch.rand(2, 784, 8, 8, 3)
    core, factors = qmf.batched_hosvd(x, rank=(49, 5, 5, 3))
    x_hat = qmf.batched_multi_mode_product(core, factors)
    (x - x_hat).abs().mean()


def test_tt_rank_upper_bounds():
    upper_bounds = qmf.tt_rank_upper_bounds([5, 20, 15, 10, 25])
    assert tuple(upper_bounds) == (5, 100, 250, 25)


def test_tt_rank_feasible_ranges():
    rank_ranges = qmf.tt_rank_feasible_ranges([5, 20, 15, 10, 25], 3)
    print(rank_ranges)


def test_ttd():
    x = torch.rand(784, 8, 8, 3)
    factors = qmf.ttd(x, rank=(49, 5, 3))
    x_hat = qmf.contract_tt(factors)
    (x - x_hat).abs().mean()


def test_batched_ttd():
    x = torch.rand(2, 784, 8, 8, 3)
    factors = qmf.batched_ttd(x, rank=(49, 5, 3))
    x_hat = qmf.batched_contract_tt(factors)
    (x - x_hat).abs().mean()


test_qmf()
test_hosvd()
test_batched_hosvd()
test_hosvd_rank_upper_bounds()
test_ttd()
test_batched_ttd()
test_tt_rank_upper_bounds()
test_tt_rank_feasible_ranges()
