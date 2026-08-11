"""k-means with an explicit balance-repair step.

Plain Lloyd converges to the data's natural cluster-mass distribution, whose
entropy is whatever it is.  Here each iteration additionally splits the q
most-loaded codewords and retires the q least-loaded ones, which trades a
little distortion for index uniformity.  Answers "can we get high entropy
*and* low residual on this data?".
"""
import numpy as np
import torch

from faithful import H, eval_assign
from mylib import DEV, K, assign, entropy_bits, load_split, t


def balanced_kmeans(X, iters=60, seed=0, q0=400, q_end=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:K]].clone()
    for it in range(iters):
        a = assign(X, C)
        cnt = torch.bincount(a, minlength=K)
        S = torch.zeros_like(C)
        S.index_add_(0, a, X)
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz].unsqueeze(1).float()
        frac = 1.0 - it / max(iters - 1, 1)
        q = int(q_end + (q0 - q_end) * frac)
        order = cnt.argsort(descending=True)
        dead = order[K - q:] if q > 0 else order[K:]
        dead = torch.cat([dead, (~nz).nonzero().squeeze(1)]).unique()
        if dead.numel():
            big = order[:dead.numel()]
            noise = torch.randn(dead.numel(), C.shape[1], generator=g,
                                device=X.device)
            noise = noise / noise.norm(dim=1, keepdim=True) * (
                0.15 * X.std())
            C[dead] = C[big] + noise
            C[big] = C[big] - noise
        if it % 15 == 0 or it == iters - 1:
            print(f"   it{it} q={q} H={entropy_bits(a):.4f}")
    return C


if __name__ == "__main__":
    V = np.load("/data/tessera-samples/cb_challenge/vectors_train.npy")
    Xt, Xv = t(V[:700000]), t(V[700000:])
    for q0 in (0, 200, 600):
        r, rv, books = Xt.clone(), Xv.clone(), []
        for lv in range(3):
            print(f"q0={q0} level {lv}")
            C = balanced_kmeans(r, iters=60, seed=lv, q0=q0)
            books.append(C)
            r = r - C[assign(r, C)]
        hs, us, resid = [], [], None
        rr = Xv.clone()
        for C in books:
            a = eval_assign(rr, C)
            hs.append(H(a))
            us.append(int((torch.bincount(a, minlength=K) > 0).sum()))
            rr = rr - C[a]
        resid = float(rr.pow(2).sum(1).mean() / Xv.pow(2).sum(1).mean())
        print(f"q0={q0}: contig-val H={[round(h,4) for h in hs]} "
              f"mean={np.mean(hs):.4f} sump={sum(2.0**h for h in hs):.1f} "
              f"resid={resid:.6g} used={us}")
        for i, C in enumerate(books):
            np.save(f"/data/tessera-samples/cb_challenge/bkm{q0}_C{i}.npy",
                    C.cpu().numpy())
