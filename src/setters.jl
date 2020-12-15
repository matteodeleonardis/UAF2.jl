function starting_point!(model::Model, data; μ::Float64 = -5.0, δλ = 10.0)
    randn!(model.x)
    model.x ./ model.L
    model.x[end] = μ
    for t in 1:model.T-1
        model.λ[t] = λmin(data.counts, t) + δλ
    end
end


function set_prior(variance::Float64, model::Model)
    var = fill(variance, fields_length(model.epistasis))
    numJ = (model.A)^2 * model.L*(model.L-1)÷2
    numh = model.A*model.L
    for l in 1:numJ
        var[l] /= numJ
    end
    for l in 1:numh
        var[numJ+l] /= numh
    end

    return GaussianPrior(var)
end


function set_optimizer(algorithm::Symbol, model::Model, data; ftol_rel = 0.0, ftol_abs = 0.0, xtol_rel = 0.0, xtol_abs = 0.0, Δλ = 1e-10)
    opt = Opt(algorithm, Model_length(model.A, model.L, model.T))
    opt.ftol_rel = ftol_rel
    opt.ftol_abs = ftol_abs
    opt.xtol_rel = xtol_rel
    opt.xtol_abs = xtol_abs
    lb = opt.lower_bounds
    lb[end-model.T+2:end] .= [λmin(data.counts, t) + Δλ for t in 1:model.T-1]
    opt.lower_bounds = lb
    return opt
end


export starting_point!, set_prior, set_optimizer
