from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, diags, identity
from scipy.sparse.linalg import inv


def RowStochastic(A: csr_matrix, strategy: str = "standard") -> csr_matrix:
    if strategy == "dmax":
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        dmax = row_sums.max()
        if dmax == 0:
            return A.copy()
        A_temp = A / dmax
        return identity(A.shape[0], format="csr") - diags(np.asarray(A_temp.sum(axis=1)).ravel()) + A_temp
    else:
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        return diags(1.0 / row_sums).dot(A)


def RecWalk(TrainSet: csr_matrix, ItemModel: csr_matrix, alpha: float = 0.01) -> csr_matrix:
    n, m = TrainSet.shape
    Muu = diags(np.ones(n))
    Mii = RowStochastic(ItemModel, "dmax")
    Hui = RowStochastic(TrainSet)
    Hiu = RowStochastic(TrainSet.transpose())
    H = vstack([hstack([csr_matrix((n, n)), Hui]), hstack([Hiu, csr_matrix((m, m))])])
    M = vstack([hstack([Muu, csr_matrix((n, m))]), hstack([csr_matrix((m, n)), Mii])])
    P = alpha * H + (1 - alpha) * M
    return P


def read_item_model(filename: str, m: int) -> csr_matrix:
    rows = []
    cols = []
    vals = []
    with open(filename, "r") as f:
        for row_idx, line in enumerate(f, start=1):
            tokens = line.strip().split()
            items = [int(tokens[i]) for i in range(0, len(tokens), 2)]
            scores = [float(tokens[i + 1]) for i in range(0, len(tokens), 2) if i + 1 < len(tokens)]
            for item, score in zip(items, scores):
                if item > 0:
                    rows.append(row_idx)
                    cols.append(item)
                    vals.append(score)
    if rows and (max(rows) < m or max(cols) < m):
        rows.append(m)
        cols.append(m)
        vals.append(0.0)
    return csr_matrix((vals, (rows, cols)), shape=(m, m))


def single_hr_rr_ndcg(pi: np.ndarray, T: np.ndarray, K: int):
    target = pi[T[0]]
    pos = np.sum(pi[T] >= target)
    if 1 <= pos <= K:
        hr = 1.0
        rr = 1.0 / pos
        ndcg = 1.0 / np.log2(pos + 1)
    else:
        hr = 0.0
        rr = 0.0
        ndcg = 0.0
    return hr, rr, ndcg
