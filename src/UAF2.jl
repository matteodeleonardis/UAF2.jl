module UAF2

using NLopt, PhageData, PhageFields, Statistics, Random, StatsBase

include("import_datasets.jl")
include("model.jl")
include("single_seq.jl")
include("likelihood.jl")
include("learn.jl")

end
