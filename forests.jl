using PythonCall, RCall
using DataFrames, StatsBase

struct ForestConfig{T <: Union{Int, Vector{Int}}}
    ntrees::T
    replicates::Integer
    mtry::Symbol # :sqrt, :log2, :all
    maxdepth::Union{Integer, Nothing}
end

ForestConfig(; ntrees::Union{Int, Vector{Int}}=500, replicates::Integer=10,
              mtry::Symbol=:sqrt, maxdepth::Union{Integer, Nothing}=nothing) =
    ForestConfig(ntrees, replicates, mtry, maxdepth)

Base.length(fc::ForestConfig{Vector{Int}}) = length(fc.ntrees)
Base.iterate(fc::ForestConfig{Vector{Int}}) =
    (ForestConfig(fc.ntrees[1], fc.replicates, fc.mtry, fc.maxdepth), 2)
Base.iterate(fc::ForestConfig{Vector{Int}}, index::Int) =
    if index <= length(fc.ntrees)
        (ForestConfig(fc.ntrees[index], fc.replicates, fc.mtry, fc.maxdepth),
         index + 1)
    end

const calc_mtry =
    Dict(:sqrt => n -> floor(Int, sqrt(n)),
         :log2 => n -> floor(Int, log2(n)),
         :third => n -> floor(Int, n/3),
         :all => n -> n)

const forest_backends = Symbol[]

struct ForestResults
    actual::Vector{Int}
    predicted::Dict{Symbol, Vector{Dict{Symbol, Any}}}
    metrics::Dict{Symbol, Vector{Dict{Symbol, Any}}}
end

# ---------------------
# General / Untility
# ---------------------

function forest_preds(data::DataFrame, depvar::Symbol, config::ForestConfig{Int}, backends::Vector{Symbol})
    X = Matrix(select(data, Not(depvar)))
    y = Int.(data[!, depvar] .== mode(data[!, depvar]))
    ForestResults(y,
                  Dict(b => forest_preds(X, y, config, b) for b in backends),
                  Dict(b => [Dict{Symbol, Any}() for _ in 1:config.replicates] for b in backends))
end

function forest_preds(X::Matrix, y::Vector{Int}, config::ForestConfig{Int}, backend::Symbol)
    printstyled("Predicting with $backend:", color=:light_black)
    start = time()
    preds = map(1:config.replicates) do i
        print(' ', i)
        forest_preds(X, y, config, Val(backend))
    end
    printstyled(" (", round(time() - start, digits=1), "s)\n", color=:light_black)
    preds
end

# ---------------------
# Scikit Learn
# ---------------------

skrandomforrest = pyimport("sklearn.ensemble" => "RandomForestClassifier")
skdecisiontree = pyimport("sklearn.tree" => "DecisionTreeClassifier")
skmetrics = pyimport("sklearn.metrics")

function forest_preds(X::Matrix, y::Vector{Int}, config::ForestConfig{Int}, ::Val{:sklearn})
    skl_params = (max_features=calc_mtry[config.mtry](size(X, 2)),
                  max_depth=if !isnothing(config.maxdepth)
                     config.maxdepth
                  else pybuiltins.None end,
                  oob_score=true)
    forest = skrandomforrest(config.ntrees; skl_params...)
    forest.fit(X, y)
    predvals = pyconvert(Matrix, forest.oob_decision_function_)[:, 2],
    treedepths = pyconvert.(Int, [dt.tree_.max_depth for dt in forest.estimators_])
    treesizes = pyconvert.(Int, [dt.tree_.node_count for dt in forest.estimators_])
    gini_imp = pyconvert(Vector, forest.feature_importances_)
    Dict(:vals => predvals,
         :treedepths => treedepths,
         :treesizes => treesizes,
         :gini_imp => gini_imp)
end

push!(forest_backends, :sklearn)

# ---------------------
# Manual Scikit Learn
# ---------------------

function bootstrapsample(r::OrdinalRange{<:Integer, <:Integer})
    inbag = rand(r, length(r))
    outofbag = setdiff(r, inbag)
    if length(outofbag) == 0
        bootstrapsample(r)
    else
        inbag, outofbag
    end
end

function forest_preds(X::Matrix, y::Vector{Int}, config::ForestConfig{Int}, ::Val{:call_sklearn})
    skl_params = (max_features=calc_mtry[config.mtry](size(X, 2)),
                 max_depth=if !isnothing(config.maxdepth)
                     config.maxdepth
                 else pybuiltins.None end)

    treedepths = Int[]
    treesizes = Int[]
    gini_imps = Matrix{Float64}(undef, size(X, 2), config.ntrees)
    predictions = zeros(Float64, size(X, 1))
    predictions_seen = zeros(Int, size(X, 1))
    for i in 1:config.ntrees
        dtree = skdecisiontree(; skl_params...)
        train, test = bootstrapsample(axes(X, 1))
        dtree.fit(X[train, :], y[train])
        preds = pyconvert(Matrix, dtree.predict_proba(X[test, :]))
        if size(preds, 2) == 1
            predictions[test] .+= y[train[1]]
        else
            predictions[test] += preds[:, 2]
        end
        predictions_seen[test] .+= 1
        gini_imps[:, i] = pyconvert(Vector, dtree.feature_importances_)
        push!(treedepths, pyconvert(Int, dtree.tree_.max_depth))
        push!(treesizes, pyconvert(Int, dtree.tree_.node_count))
    end
    Dict(:vals => predictions ./ predictions_seen,
         :treedepths => treedepths,
         :treesizes => treesizes,
         :gini_imp => mean(gini_imps, dims=2))
end

push!(forest_backends, :call_sklearn)

# ---------------------
# R's randomForest
# ---------------------

R"library(randomForest)"

function forest_preds(X::Matrix, y::Vector{Int}, config::ForestConfig{Int}, ::Val{:randomForest})
    @rput X y
    R"""rf <- randomForest(
            X, factor(y),
            ntree = $(config.ntrees),
            mtry = $(calc_mtry[config.mtry](size(X, 2))),
            maxnodes = $(if !isnothing(config.maxdepth) 1+config.maxdepth else R"NULL" end)
        )"""
    Dict(:vals => R"rf$votes[,2]" |> rcopy,
         :treesizes => R"treesize(rf)" |> rcopy,
         :gini_imp => R"rf$importance[,1]" |> rcopy)
end

push!(forest_backends, :randomForest)

# ---------------------
# ranger
# ---------------------

R"library(ranger)"

function forest_preds(X::Matrix, y::Vector{Int}, config::ForestConfig{Int}, ::Val{:ranger})
    @rput X y
    R"Xy <- cbind(X, y)"
    R"""rrf -> ranger::ranger(
            data = Xy,
            dependent.variable.name = "y",
            num.trees = $(config.ntrees),
            mtry = $(calc_mtry[config.mtry](size(X, 2))),
            max.depth = $(something(config.maxdepth, 0)),
            probability = TRUE,
            replace = TRUE,
            classification = TRUE)$predictions[,2]
    """ |> rcopy
    Dict(:vals => R"rrf$predictions[,2]" |> rcopy,
         :treesizes => R"sapply(1:rrf$num.trees, function(n) {dim(treeInfo(rrf, tree=n))[1]})" |> rcopy,
         :gini_imp => R"rrf$variable.importance" |> rcopy)
end

push!(forest_backends, :ranger)
