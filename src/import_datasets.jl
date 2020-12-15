function import_dataset(name::Symbol)
    if name == :boyer
        return boyer2016pnas(:f3pvp,:aa; gaps=false)
    elseif name == :fowler
        return fowler2010nmeth(:aa; gaps=false)
    elseif name ==:araya
        return araya2012pnas(:aa,gaps=false)
    elseif name == :wu
        return wu2016elife(:aa)
    elseif name == :olson
        return olson2014currbio_gb1()
    else
        return println("dataset not found")
    end
end

export import_dataset
