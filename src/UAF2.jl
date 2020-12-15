module UAF2

using NLopt, PhageData, PhageFields, Statistics, Random, StatsBase

include("log_approx.jl")
include("import_datasets.jl")
include("model.jl")
include("single_seq.jl")
include("likelihood.jl")
include("preprocess_utils.jl")
include("setters.jl")
include("learn.jl")

end
