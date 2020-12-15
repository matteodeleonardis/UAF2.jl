function not_disappearing(data)
    idx=[]
    for s in 1:data.S
        flag = true
        for t in 1:data.T-1
            if data.counts[s,1,t]==0 && data.counts[s,1,t+1]>0
                flag = false
                break
            end
        end

        if flag
            push!(idx,s)
        end
    end
    return idx
end


function filter_counts(data, rounds::Vector{Int}, counts_threshold::Int)
    @assert maximum(rounds) <= data.T
    @assert length(rounds) <= data.T

    data_filtered = deepcopy(data)
    for t in rounds
        data_filtered = subdata(data, findall(data.counts[:,1,t] .>= counts_threshold))
    end
    return data_filtered
end


function add_pseudocounts(data, pc)
    data_new = deepcopy(data)
    data_new.counts .+= pc
    return data_new
end


export not_desappearing, filter_counts, add_pseudocounts
