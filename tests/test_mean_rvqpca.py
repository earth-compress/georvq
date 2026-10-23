"""RVQ + PCA mean channel: fixed record size, exact pack round trip,
decode determined by the codes, and error falling with the budget."""
import numpy as np

from georvq.inference.mean_codec import RVQPCAMeanChannel


def _data(n=4000, d=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(6, d)) * 3
    x = centres[rng.integers(0, 6, n)] + rng.normal(size=(n, d)) * 0.3
    return x.astype(np.float32)


def _books(x, L=2, K=16, seed=0):
    rng = np.random.default_rng(seed)
    r, books = x.copy(), []
    for _ in range(L):
        C = r[rng.choice(len(r), K, replace=False)].copy()
        for _ in range(10):
            a = ((r[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            for k in range(K):
                if (a == k).any():
                    C[k] = r[a == k].mean(0)
        a = ((r[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        r = r - C[a]
        books.append(C)
    return np.stack(books), r


def test_fixed_cost_and_pack_round_trip(tmp_path):
    x = _data()
    books, resid = _books(x)
    ch = RVQPCAMeanChannel.fit_residual(books, resid, total_bits=40)
    assert ch.cost_bits() == 40 and ch.record_bytes() == 5
    f = ch.encode(x)
    data = ch.pack(f)
    assert data.shape == (len(x), 5)
    assert (ch.unpack(data) == f).all()
    ch.save(tmp_path / "m.npz")
    ch2 = RVQPCAMeanChannel.load(tmp_path / "m.npz")
    assert np.array_equal(ch2.decode(ch2.unpack(data)), ch.decode(f))


def test_error_falls_with_budget():
    x = _data()
    books, resid = _books(x)
    errs = []
    for B in (8, 24, 64):
        ch = RVQPCAMeanChannel.fit_residual(books, resid, total_bits=B)
        errs.append(float(np.abs(ch.round_trip(x) - x).mean()))
    assert errs[0] > errs[1] > errs[2]


def _fit_coded(x, total_bits, pre_scale=None, pre_log=None):
    """Books and PCA coder fitted in the transform's coded space."""
    from georvq.inference.mean_codec import coded_space
    y = coded_space(x, pre_scale, pre_log)
    books, resid = _books(y)
    return RVQPCAMeanChannel.fit_residual(books, resid, total_bits=total_bits,
                                          pre_scale=pre_scale,
                                          pre_log=pre_log)


def test_untransformed_file_has_no_transform(tmp_path):
    x = _data()
    books, resid = _books(x)
    ch = RVQPCAMeanChannel.fit_residual(books, resid, total_bits=40)
    ch.save(tmp_path / "m.npz")
    assert sorted(np.load(tmp_path / "m.npz").files) == \
        ["V", "bits", "books", "clip", "mu0", "s"]
    assert len(ch.id_state()) == 6
    assert ch.pre_scale is None and ch.pre_log is None


def test_transform_persists_and_decodes_alike(tmp_path):
    x = _data()
    x[:, 4:] = np.exp(x[:, 4:] * 0.3)          # positive, log-coded
    scale = np.linspace(0.5, 4.0, x.shape[1]).astype(np.float32)
    lg = np.arange(x.shape[1]) >= 4
    ch = _fit_coded(x, 48, scale, lg)
    f = ch.encode(x)
    ch.save(tmp_path / "m.npz")
    from georvq.inference.mean_codec import load_channel
    ch2 = load_channel(tmp_path / "m.npz")
    assert np.array_equal(ch2.pre_scale, scale)
    assert np.array_equal(ch2.pre_log, lg)
    assert np.array_equal(ch2.encode(x), f)
    assert np.array_equal(ch2.decode(f), ch.decode(f))
    assert (ch.decode(f)[:, lg] > 0).all()
    assert len(ch.id_state()) == 8
    # the transform is part of what a decoder must agree on
    other = RVQPCAMeanChannel(ch.books, ch.mu0, ch.V, ch.s, ch.bits, ch.clip,
                              scale * 2, lg)
    assert not np.array_equal(other.decode(f), ch.decode(f))


def test_scale_moves_error_between_dimensions():
    x = _data(d=6)
    w = np.ones(6, np.float32)
    w[0] = 20.0                               # dimension 0 costs 400x more
    plain = _fit_coded(x, 24, np.ones(6, np.float32))
    heavy = _fit_coded(x, 24, w)
    e_plain = np.abs(plain.round_trip(x) - x).mean(0)
    e_heavy = np.abs(heavy.round_trip(x) - x).mean(0)
    assert e_heavy[0] < 0.5 * e_plain[0]
    assert e_heavy[1:].mean() > e_plain[1:].mean()


def test_log_dimensions_code_relative_error():
    rng = np.random.default_rng(1)
    # four decades of magnitude: a log-coded dimension keeps the relative
    # error flat across them, a linear one does not
    x = np.exp(rng.uniform(0, np.log(1e4), (4000, 3))).astype(np.float32)
    ch = _fit_coded(x, 36, np.ones(3, np.float32), np.ones(3, bool))
    rel = np.abs(ch.round_trip(x) / x - 1)
    small, large = x < 10, x > 1000
    assert rel[small].mean() < 2 * rel[large].mean()
    assert rel[large].mean() < 2 * rel[small].mean()
    import pytest
    with pytest.raises(ValueError):
        ch.encode(-x)
