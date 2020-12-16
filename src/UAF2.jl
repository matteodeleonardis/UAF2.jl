module UAF2

using NLopt, PhageData, PhageFields, Statistics, Random, StatsBase, PyPlot

include("log_approx.jl")
include("import_datasets.jl")
include("model.jl")
include("single_seq.jl")
include("likelihood.jl")
include("preprocess_utils.jl")
include("setters.jl")
include("learn.jl")
include("validation_utils.jl")
include("likelihood_rare.jl")
include("learn_rare.jl")

    module BioseqUtils_dev
        include("BioseqUtils_dev.jl")
    end

end
