function Likleihood_fields_rare(model::Model, data)
	L = 0.0
	@inbounds @simd for s in 1:data.S
		Hs = Energy(model.epistasis, model.x, data.variants[s])
		m = ms(data.counts[s,1,:], model.λ)
		M = Ms(data.counts[s,1,:])
		L += -M*exp(-Hs) - m*Hs
	end
	return L
end

function gradient_Likelihood_fields_rare(model::Model, data)
	grad = zeros(length(model.x))
	@inbounds for s in 1:data.S
		Hs = Energy(model.epistasis, model.x, data.variants[s])
		m = ms(data.counts[s,1,:], model.λ)
		M = Ms(data.counts[s,1,:])
		g = m - exp(-Hs)*M

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

function Likelihood_rare(model::Model, data)
	L = 0.0
	@inbounds H = [Energy(model.epistasis, model.x, data.variants[s]) for s in 1:data.S]
	@inbounds for t in 1:model.T-1
		@simd for s in 1:data.S
			L += ( log_binom(data.counts[s,1,t], data.counts[s,1,t+1]/model.λ[t])
					- H[s]*data.counts[s,1,t+1]/model.λ[t] - exp(-H[s])*data.counts[s,1,t] )
		end
		L -= data.S*log(model.λ[t])
	end
	return L
end

function gradient_Likelihood_rare(model::Model, data)
	gf = gradient_Likelihood_fields_rare(model, data)
	gl = gradient_Likelihood_lambda(model, data)
	return vcat(gf, gl)
end
