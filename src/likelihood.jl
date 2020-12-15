#computes ms that appears in the part of the loglikelihood that depends on model parameters
function ms(Ns::Vector{Float64}, λ::Vector{Float64}) #Ns are counts of variant s throut the various rounds
	m = 0.0
	@inbounds @simd for t in 1:length(Ns)-1
		m += Ns[t+1]/λ[t]
	end
	return m
end


#computes Ms that appears in the part of the loglikelihood that depends on model parameters
function Ms(Ns::Vector{Float64})
	M = 0.0
	@inbounds @simd for t in 1:length(Ns)-1
		M += Ns[t]
	end
	return M
end


#computes the minimum value that λ(t) can assume consistent with the data
function λmin(N::Array{Float64,3},t::Int)
	(S, F, T) = size(N)
	@assert t > 0 && t < T
	l = -Inf
	@inbounds for s in 1:S
		if N[s,1,t] == 0
			continue
		end
		l = (l > N[s,1,t+1]/N[s,1,t]) ? l : N[s,1,t+1]/N[s,1,t]
	end
	return l
end


#computes the value of the term of the loglikelihood that depends only on model parameters
function Likelihood_fields(model::Model, data)
	L = 0.0
	@inbounds @simd for s in 1:data.S
		Hs = Energy(model.epistasis, model.x, data.variants[s])
		m = ms(data.counts[s,1,:], model.λ)
		M = Ms(data.counts[s,1,:])
		L += -M*log_oneplusexp(Hs) + (M-m)*Hs
	end
	return L
end


#computes the gradient with respect to model parameters of the loglikelihood
function gradient_Likelihood_fields(model::Model, data)
	grad = zeros(length(model.x))
	@inbounds for s in 1:data.S
		p = pb(model.epistasis, model.x, data.variants[s])
		m = ms(data.counts[s,1,:], model.λ)
		M = Ms(data.counts[s,1,:])
		g = m - p*M

		@simd for i in 1:data.L
			index = field_index(model.epistasis, :h, data.variants[s].sequence[i], i)
			grad[index] += g
		end
		for i in 1:data.L-1
			@simd for j in i+1:data.L
				index = field_index(model.epistasis, :J, data.variants[s].sequence[i], data.variants[s].sequence[j], i, j)
				grad[index] += g
			end
		end
		grad[length(model.x)] += g
	end
	return grad
end


#computes the part of the loglikelihood that depends on λ's'
function Likelihood_lambda(model::Model, data)
	L = 0.0
	@inbounds H = [Energy(model.epistasis, model.x, data.variants[s]) for s in 1:data.S]
	@inbounds for t in 1:data.T-1
		@simd for s in 1:data.S
			L += log_binom(data.counts[s,1,t], data.counts[s,1,t+1]/model.λ[t]) - H[s]*data.counts[s,1,t+1]/model.λ[t]
		end
		L -= data.S*log(model.λ[t])
	end
	return L
end


#computes the gradient of the loglikelihood with respect to λ's'
function gradient_Likelihood_lambda(model::Model, data)
	grad = zeros(model.T-1)
	@inbounds H = [Energy(model.epistasis, model.x, data.variants[s]) for s in 1:data.S]
	@inbounds for t in 1:model.T-1
		@simd for s in 1:data.S
			grad[t] += ( xlogy(data.counts[s,1,t+1]/model.λ[t]^2, data.counts[s,1,t+1]/model.λ[t])
						- xlogy(data.counts[s,1,t+1]/model.λ[t]^2, data.counts[s,1,t]-data.counts[s,1,t+1]/model.λ[t])
						+ H[s]*data.counts[s,1,t+1]/model.λ[t]^2)
		end
		grad[t] -= data.S/model.λ[t]
	end
	return grad
end


#computes the value of the total loglikelihood
function Likelihood(model::Model, data)
	L = 0.0
	@inbounds H = [Energy(model.epistasis, model.x, data.variants[s]) for s in 1:data.S]
	@inbounds for t in 1:model.T-1
		@simd for s in 1:data.S
			L += ( log_binom(data.counts[s,1,t], data.counts[s,1,t+1]/model.λ[t])
					- H[s]*data.counts[s,1,t+1]/model.λ[t] - log_oneplusexp(-H[s])*data.counts[s,1,t] )
		end
		L -= data.S*log(model.λ[t])
	end
	return L
end


#computes the gradient of the loglikelihood with respect to all the parameters that must be learnt
#first the parameters of the model and then the λ's'
function gradient_Likelihood(model::Model, data)
	gf = gradient_Likelihood_fields(model, data)
	gl = gradient_Likelihood_lambda(model, data)
	return vcat(gf, gl)
end
