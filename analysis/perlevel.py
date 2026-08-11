exec(open('/data/tessera-samples/cb_private/fit_sweep.py').read().split("print(\"=== 1.")[0])
print(f"w32hd fine tier dim {D}, fit {FIT.shape[0]}, eval {EV.shape[0]}")
R = FIT.clone(); books = []
for _ in range(8):
    C = km(R, 16384); books.append(C); R = R - C[_chunked_assign(R, C)]
R = EV.clone(); prev = 1.0
print(f"{'lvl':>4} {'resid E/E0':>11} {'this lvl':>9} {'bits(emp)':>10} {'perplex':>8}")
for i, C in enumerate(books):
    a = _chunked_assign(R, C)
    cnt = torch.bincount(a, minlength=16384).float(); p = cnt/cnt.sum()
    H = float(-(p[p>0]*p[p>0].log2()).sum())
    perp = float(torch.exp(-(p[p>0]*p[p>0].log()).sum()))
    R = R - C[a]
    cur = float(R.pow(2).sum(-1).mean())/e0
    print(f"{i:>4} {cur:>11.4f} {cur/prev:>9.3f} {H:>10.2f} {perp:>8.0f}")
    prev = cur
