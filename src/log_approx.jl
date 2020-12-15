function log_oneplusexp(E::Float64)
    return (E < 50) ? log(1+exp(E)) : E
end

function xlogy(x::Float64, y::Float64)
    if x == 0.0
        return 0.0
    else
        return x*log(y)
    end
end


function log_binom(n::Float64, k::Float64)
    @assert n >= k
    if (n == 0) || (k == 0)
       return 0
    end
    return n*log(n)-k*log(k)-(n-k)*log(n-k)
end
