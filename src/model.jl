struct Model{A,L}
	epistasis::EpistasisMu{A,L}
	x::Vector{Float64}
	λ::Vector{Float64}
	A::Int
	L::Int
	T::Int
end


#creates a model for epistasis
function Model(epistasis::EpistasisMu{A,L}, T::Int, x = [], λ = []) where {A,L}
	if length(x) == 0
		x = zeros(fields_length(epistasis))
	end
	if length(λ) == 0
		λ = zeros(T-1)
	end
	return Model(epistasis, x, λ, A, L, T)
end


#returns the total number of parameters to learn
function Model_length(A::Int, L::Int, T)
	return A^2*L*(L-1)÷2 + A*L + 1 + T -1
end


export Model, Model_length
