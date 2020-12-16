include("utilities.jl")


#selectivity curve using spearman correlation
function spearman_selectivity_filters_curves( data, predictions::Vector{Vector{Float64}};
    plot::Bool=true,
    selectivity=data.θ,
    filter_feature=-data.Δ,
    thresh=compute_thresh(filter_feature),
    labels=[string(i) for i=1:length(predictions)])

    Sf=length(filter_feature)
    n_predictions=size(predictions)[1]

    v_thr_cor=zeros(length(thresh),n_predictions+1)
    for t=1:length(thresh)

            filter_idx = [s for s = 1 : Sf if filter_feature[s] ≥ thresh[t]];

            if !isempty(filter_idx)
                v_thr_cor[t,1] = length(filter_idx)/Sf;
                for k=1:n_predictions
                    v_thr_cor[t,2] = corspearman(predictions[k][filter_idx],selectivity[filter_idx])
                end
            end
    end

    if plot

        for k=1:n_predictions
            semilogx(v_thr_cor[:,1],v_thr_cor[:,k+1],label=labels[k])
        end
        ylabel("Spearman correlation")
        xlabel("fraction of sequences")
        legend()
        title("correlation with mean selectivity")
    else
        return v_thr_cor
    end

end


#plots logselectivity vs energy curves for all train-set splits of the crossvalidation
function plot_fold_curves(data::Vector{Any}, energies::Vector{Vector{Float64}}, range=log_range(1/100,1,20); metric::Symbol = :pearson,
        xlab::String, ylab::String, tit::String)
    nfolds = length(data)
    sel_curves = (metric == :pearson) ? selectivity_filters_curves : spearman_selectivity_filters_curves
    corr_train = [sel_curves(data[k][1], [-energies[k]],
        thresh = compute_thresh(-data[k][1].Δ;
        fraction=range), plot = false) for k in 1:nfolds];
    for k in 1:nfolds
        plot(corr_train[k][:,1], corr_train[k][:,2])
    end
    xlabel(xlab)
    ylabel(ylab)
    title(tit)
    xscale(:log)
    return corr_train
end


export spearman_selectivity_filters_curves, plot_fold_curves 
