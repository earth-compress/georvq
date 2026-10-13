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
