"""진단: 원통으로 안 잡힌 '휜' 영역(작은 조각이 매끈하게 이어진 덩어리)을 보여 준다."""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recover  # noqa: E402

for f in sys.argv[1:]:
    V, F = recover.load_stl(f)
    m = recover.Mesh(V, F, 0.01)
    m.patches()
    c = m.cylinders()
    used = set().union(*[x["patches"] for x in c]) if c else set()
    N = m.N
    ta, tb = m.et[:, 0], m.et[:, 1]
    free = ~np.isin(m.pid, list(used))
    dih = np.arccos(np.clip(np.einsum("ij,ij->i", N[ta], N[tb]), -1, 1))
    j = free[ta] & free[tb] & (m.pid[ta] != m.pid[tb]) & (dih > recover.ANG_TOL) & (dih <= recover.THETA_MAX)
    comp = recover._components(len(F), ta[j], tb[j])
    inv = np.zeros(len(F), bool)
    inv[ta[j]] = inv[tb[j]] = True
    print(os.path.basename(f), "원통", len(c))
    for cc, n in Counter(comp[inv]).most_common(12):
        tt = np.nonzero(inv & (comp == cc))[0]
        vs = np.unique(F[tt])
        nn = N[tt]
        ev = np.linalg.eigvalsh(nn.T @ nn)
        print(f"  삼각형 {n:4d} 조각크기 {dict(Counter(m.ptri_count[m.pid[tt]].tolist()))} "
              f"중심 {np.round(V[vs].mean(0), 1)} 법선고유값 {np.round(ev / ev.sum(), 3)}")
