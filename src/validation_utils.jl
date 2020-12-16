#creates a random partition of 1:S in n_folds folds
function create_folds(S::Int, n_folds::Int)
    samples_per_fold = S ÷ n_folds
    fold_samples = fill(samples_per_fold, n_folds-1) #temporary
    push!(fold_samples, S - samples_per_fold*(n_folds-1))
    idx_folds = randsplit(S, fold_samples...)
end


#creates train and test set where k is the fold to be used as test
function traintest_split(data, ind_folds::Vector, k::Int)
    ind_test = ind_folds[k]
    ind_train = vcat(ind_folds[1:end .!= k]...)
    data_test = subdata(data, ind_test)
    data_train = subdata(data, ind_train)
    return (data_train, ind_train), (data_test, ind_test)
end


#performs the entire crossvalidation and writes in the supplied variables the learning result
function crossvalidation!(data, model::Model, prior_var::Float64, nfolds::Int, training_set::Vector{Any}, test_set::Vector{Any}, model_parameters::Vector{Any},
     outcomes::Vector{Any}, L2reg::Vector{Any}, logP::Vector{Any}, obs::Vector{Any}, verbose::Bool=true; ftol_rel = 1e-7, Δλ = 1e-10, monitor = nothing)

     ind_folds = create_folds(data.S, nfolds)

    for k in 1:nfolds
        if verbose
            println("fold ", k, "/", nfolds, "    ************************")
        end

        #creating test set and training set
        (data_train, ind_train), (data_test, ind_test) = traintest_split(data, ind_folds, k)

        #setting starting point for learning
        starting_point!(model, data_train)

        #setting up regularization
        prior = set_prior(prior_var, model)

        #learning
        opt = set_optimizer(:LD_MMA, model, data_train, ftol_rel = ftol_rel, Δλ = Δλ)
        model, ret, loglike, prop = learn!(model, prior, opt, data_train; monitor = monitor)

        push!(training_set, (data_train, ind_train))
        push!(test_set, (data_test, ind_test))
        push!(model_parameters, model)
        push!(outcomes, ret)
        push!(L2reg, prior)
        push!(logP, loglike)
        push!(obs, prop)
    end
    return nothing
end


#computes the average curve and associate to it an error using the standard deviation of the mean
function avg_selectivity_curve(θcurves_train)
    avg_corr_train = Array{Float64, 2}(undef,size(θcurves_train[1]))

    for j in 1:size(avg_corr_train)[1]
        avg_corr_train[j,1] = θcurves_train[1][j,1]
        avg_corr_train[j,2] = mean([θcurves_train[k][j,2] for k in 1:length(θcurves_train)])
    end

    std_corr_train = Array{Float64, 1}(undef, size(avg_corr_train)[1])

    for j in 1:length(std_corr_train)
        std_corr_train[j] = std([θcurves_train[k][j,2] for k in 1:length(θcurves_train)])
    end
    return avg_corr_train, std_corr_train ./ sqrt(length(θcurves_train))
end


#computes energies and probabilities of a dataset using learnt parameters
function inference(model::Model, data)
    energies = [Energy(model.epistasis, model.x, s) for s in data.variants]
    p = [pb(model.epistasis, model.x, s) for s in data.variants]
    return (energies, p)
end


#computes energies and probabilities of a dataset using learnt parameters for all train-test instancies of the crossvalidation
function traintest_inference(model_parameters::Vector{Any}, data_train::Vector{Any}, data_test::Vector{Any})
    nfolds = length(model_parameters)
    etr = Vector{Vector{Float64}}(undef, nfolds)
    ptr = Vector{Vector{Float64}}(undef, nfolds)
    ete = Vector{Vector{Float64}}(undef, nfolds)
    pte = Vector{Vector{Float64}}(undef, nfolds)

    for k in 1:nfolds
    etr[k], ptr[k] = inference(model_parameters[k], data_train[k][1])
    ete[k], pte[k] = inference(model_parameters[k], data_test[k][1])
    end
    return (etr, ptr), (ete, pte)
end


#plots the probability distribution as function of energies for a dataset
function plot_distribution(energies::Vector{Float64}, ps::Vector{Float64})
    ind = sortperm(energies, rev=true)
    plot(energies[ind], ps[ind])
    xlabel("Es")
    ylabel("ps")
end


#plots Ns(t) vs Ns(t+1)/λ(t) to check for deterministic binding regime
function deterministic_binding(pb::Vector{Float64}, λ::Float64, data, t::Int; count_thr = 20)
    ind = findall(data.counts[:,1,t] .> count_thr);
    average = pb[ind] .* data.counts[ind,1,t]
    variance = pb[ind] .* (1.0 .- pb[ind]) .* data.counts[ind,1,t]
    line = LinRange(0.0, max(maximum(data.counts[ind,1,t+1] ./ λ), maximum(average)),10)
    errorbar(data.counts[ind,1,t+1] ./ λ, average, sqrt.(variance), ls="None", marker="s")
    corr11 = cor(data.counts[ind,1,t+1] ./ λ, average)

    plot(line,line, linewidth=1, color="red")
    title("(t=$t)   corr= $corr11")
    ylabel("N(s)ps")
    xlabel("Ns(t+1)/λ(t)")
end


#does the same of the previous function but plots the result in a logarithmic scale
function log_deterministic_binding(pb::Vector{Float64}, λ::Float64, data, t::Int; count_thr = 20)
    ind = findall(data.counts[:,1,t] .> count_thr);
    average = pb[ind] .* data.counts[ind,1,t]
    variance = pb[ind] .* (1.0 .- pb[ind]) .* data.counts[ind,1,t]
    line = LinRange(0.0, max(maximum(data.counts[ind,1,t+1] ./ λ), maximum(average)),10)
    xscale(:log)
    yscale(:log)
    errorbar(data.counts[ind,1,t+1] ./ λ, average, sqrt.(variance), ls="None", marker="s")
    corr11 = cor(data.counts[ind,1,t+1] ./ λ, average)

    plot(line,line, linewidth=1, color="red")
    title("(t=$t)   corr= $corr11")
    ylabel("N(s)ps")
    xlabel("Ns(t+1)/λ(t)")
end




export crossvalidation!, avg_selectivity_curve, inference, traintest_inference,
    plot_distribution, deterministic_binding, log_deterministic_deterministic
