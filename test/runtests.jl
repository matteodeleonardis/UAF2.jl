ENV["PHAGEDATAPATH"]="/home/matteo/poliTo/Tesi/Data"

using UAF2, PhageFields, PhageData, NLopt
using Test

boyer = import_dataset(:boyer)

boyer = subdata(boyer, not_disappearing(boyer))
boyer = add_pseudocounts(boyer, 0.5)

epistasis = EpistasisMu{boyer.A, boyer.L}()
model = Model(epistasis, boyer.T)

ntrain = round(Int, boyer.S*0.8)
train, test = randtrain(boyer, ntrain)

starting_point!(model, train)
prior = set_prior(100.0, model)
opt = set_optimizer(:LD_MMA, model, train; ftol_rel = 1e-7, Δλ = 1e-10)

get_lambda(m ,p ,d) = m.λ
get_mu(m, p, d) = [m.x[end]];

model, ret, logP, properties = learn!(model, prior, opt, train, monitor = [get_lambda, get_mu])
