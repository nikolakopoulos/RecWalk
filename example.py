from __future__ import annotations
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import inv

from recwalk import RecWalk, read_item_model, single_hr_rr_ndcg


def main():
    TopN = 10

    data = loadmat("yahoo.mat")
    TrainSet = csr_matrix(data["TrainSet"])
    Holdout = data["Holdout"]
    UW = data["SampledUnwatched"]
    n, m = TrainSet.shape

    W = read_item_model("example.model", m)
    P = RecWalk(TrainSet, W, alpha=0.005)

    HR = np.zeros(n)
    RR = np.zeros(n)
    NDCG = np.zeros(n)

    PI = TrainSet.dot(W)
    for user in range(n):
        items = np.concatenate([Holdout[user], UW[:, user]])
        HR[user], RR[user], NDCG[user] = single_hr_rr_ndcg(
            PI.getrow(user).toarray().ravel(), items, TopN
        )
    print(f"Base Item Model:  HR={HR.mean()}  ARHR={RR.mean()}  NDCG={NDCG.mean()}")

    K = 7
    for user in range(n):
        ru = csr_matrix(P.getrow(user))
        for _ in range(2, K + 1):
            ru = ru.dot(P)
        items = np.concatenate([Holdout[user], UW[:, user]])
        HR[user], RR[user], NDCG[user] = single_hr_rr_ndcg(
            ru[:, n:].toarray().ravel(), items, TopN
        )
    print(f"RecWalk K-Step:   HR={HR.mean()}  ARHR={RR.mean()}  NDCG={NDCG.mean()}")

    eta = 0.7
    I = identity(P.shape[0], format="csr")
    PI = inv(I - eta * P).toarray()
    PI = PI[:n, n:]
    for user in range(n):
        items = np.concatenate([Holdout[user], UW[:, user]])
        HR[user], RR[user], NDCG[user] = single_hr_rr_ndcg(
            PI[user, :], items, TopN
        )
    print(f"RecWalk PR:       HR={HR.mean()}  ARHR={RR.mean()}  NDCG={NDCG.mean()}")


if __name__ == "__main__":
    main()
