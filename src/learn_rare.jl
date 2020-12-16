function learn_rare!(model::Model, prior::GaussianPrior, opt::Opt, data; monitor = nothing)
	λm = [λmin(data.counts, t) for t in 1:model.T-1]
	for t in 1:model.T-1
		@assert model.λ[t] >= λm[t]
	end
	xinit = vcat(model.x, model.λ)
	logP = []
	properties = [[] for a in monitor]
	function f(x::Vector, grad::Vector, vals::Vector, prop::Vector)
		model.x .= x[1:end-model.T+1]
		model.λ .= x[end-model.T+2:end]
		if length(grad) > 0
			grad .= gradient_Likelihood_rare(model, data)
			@inbounds prior(grad[1:end-model.T+1], x[1:end-model.T+1])
		end
		l = Likelihood_rare(model, data) + prior(x[1:end-model.T+1])
		push!(vals, l)
		if monitor != nothing
			for i in 1:length(monitor)
				push!(prop[i], monitor[i](model, prior, data)[:])
			end
		end
		return l
	end

	f1(x::Vector, grad::Vector) = f(x, grad, logP, properties)
	opt.max_objective = f1
	(maxf, xmax, ret) = optimize(opt, xinit)
	@inbounds model.x .= xmax[1:end-model.T+1]
	@inbounds model.λ .= xmax[end-model.T+2:end]
	return (model, ret, logP, properties)
end
