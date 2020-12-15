#computes the energy of a sequence
#energy(epistasis, parameters, sequence)
function Energy(epistasis::EpistasisMu, x::Vector{Float64}, variant)
	return energy(epistasis, x, variant)
end


#computes the binding probability of a sequence
#pb(epistasis, parameters, sequence)
function pb(epistasis::EpistasisMu, x::Vector{Float64}, variant)
	E = Energy(epistasis, x, variant)
	return 1/(1+exp(E))
end


export Energy, pb
