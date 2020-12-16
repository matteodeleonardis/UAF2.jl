#creates a logarithmic range
function log_range(start, stop, points)
    r = LinRange(log(start), log(stop), points)
    r = exp.(r)
    return r
end


export log_range
