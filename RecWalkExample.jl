using Distributed
addprocs()

@everywhere using SharedArrays
@everywhere using LinearAlgebra
@everywhere using SparseArrays

using MAT

include("include.jl")

TopN = 10


# Read Data
DATA = matread("yahoo.mat")
TrainSet = DATA["TrainSet"]
Holdout = DATA["Holdout"]
UW = DATA["SampledUnwatched"]
n, m = size(TrainSet)

# Read Item Model
W = readItemModel("example.model",m)

# Build RecWalk Model
P = RecWalk(TrainSet,W,0.005)

NDCG = SharedArray{Float64}(n)
RR = SharedArray{Float64}(n)
HR = SharedArray{Float64}(n)

#Base Item Model
PI = TrainSet*W
@sync @distributed for user = 1:n
	HR[user], RR[user], NDCG[user] = Single_HR_RR_NDCG(PI[user,:], vcat(Holdout[user], UW[:,user]), TopN)
end
println("Base Item Model:  HR = $(mean(HR))  ARHR=$(mean(RR))  NDCG=$(mean(NDCG))")


# RecWalk - K-Step
K = 7
@sync @distributed for user = 1:n
    ru  = sparse(reshape(P[user,:], 1, m+n))
    [ru  *= P for step=2:K]
    HR[user], RR[user], NDCG[user] = Single_HR_RR_NDCG(ru[n+1:end], vcat(Holdout[user], UW[:,user]), TopN)
end
println("RecWalk K-Step:   HR = $(mean(HR))  ARHR=$(mean(RR))  NDCG=$(mean(NDCG))")

# RecWalk - PR
eta = 0.7
PI = inv(full(I-eta*P)) # due to the small size of the example data the recwalk ppr vectors can be computed in batch.
PI = PI[1:n,n+1:end]
@sync @distributed for user = 1:n
	HR[user], RR[user], NDCG[user] = Single_HR_RR_NDCG(PI[user,:], vcat(Holdout[user], UW[:,user]), TopN)
end
println("RecWalk PR:       HR = $(mean(HR))  ARHR=$(mean(RR))  NDCG=$(mean(NDCG))")





