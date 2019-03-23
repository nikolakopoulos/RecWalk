function RecWalk(TrainSet, ItemModel, α=0.01)
    n,m = size(TrainSet)
    Muu = speye(n)
    Mii = RowStochastic(ItemModel,"dmax")
    Hui = RowStochastic(TrainSet)
    Hiu = RowStochastic(TrainSet')
    H = vcat(hcat(spzeros(n,n),Hui), hcat(Hiu,spzeros(m,m)))
    M = vcat(hcat(Muu,spzeros(n,m)), hcat(spzeros(m,n),Mii))
    P = α*H+(1-α)*M
    return P
end

function RowStochastic(A, strategy="standard")
    if strategy == "dmax"
        row_sums = sum(A, 2)
        dmax = maximum(row_sums)
        A_temp = 1 / dmax * A
        return (I - spdiagm(vec(sum(A_temp, 2)))) + A_temp 
    else
        row_sums = sum(A, 2)
        row_sums[sum(A, 2) .== 0] = 1 # replacing the zero row sums with 1
        return spdiagm(vec(1 ./ row_sums)) * A
    end

end

function readItemModel(filename, m)
    A = readdlm(filename, skipblanks = false)
    A[A .== ""] = 0; A = Array{Float64}(A) 
    s1, s2 = size(A)
    rows = []; cols = []; vals = Float64[];
    for i = 1:s1
        items = Array{Int64}(A[i,1:2:s2 - 1]) 
        indx = find(items .> 0) 
        items = items[indx]
        scores =  A[i,2 * indx] 
        append!(rows, i * ones(length(items)))
        append!(cols, items)
        append!(vals, scores)
    end
    if maximum(rows) < m || maximum(cols) < m
        append!(rows, [m])
        append!(cols, [m])
        append!(vals, [0])
    end
    return sparse(rows, cols, vals)
end

@everywhere function Single_HR_RR_NDCG(Π, T, K)
    target = Π[T[1]]
    pos = sum(Π[T].>=target) 
    if 1 <= pos <= K
            HR = 1; RR = 1 / pos; NDCG = 1/log2(pos+1) 
    else
            HR = 0; RR = 0; NDCG = 0
    end
    return HR, RR, NDCG
end



