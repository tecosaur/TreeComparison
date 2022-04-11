include("forests.jl")

skmetrics = pyimport("sklearn.metrics")

function metric_precrec!(res::ForestResults)
    for backend in keys(res.predicted)
        for i in 1:length(res.predicted[backend])
            precision, recall, precision_recall_thresholds =
                pyconvert.(Vector, skmetrics.precision_recall_curve(res.actual, res.predicted[backend][i]))
            f1scores = replace((@. 2*recall*precision / (recall+precision)), NaN => 0.0)
            res.metrics[backend][i][:precision] = precision
            res.metrics[backend][i][:recall] = recall
            res.metrics[backend][i][:precision_recall_thresholds] = precision_recall_thresholds
            res.metrics[backend][i][:auPRC] = pyconvert(Float64, skmetrics.auc(recall, precision))
            res.metrics[backend][i][:f1max] = maximum(f1scores)
        end
    end
end

function metric_roc!(res::ForestResults)
    for backend in keys(res.predicted)
        for i in 1:length(res.predicted[backend])
            αs, πs, απ_thresholds =
                pyconvert(Vector{Vector{Float64}},
                          skmetrics.roc_curve(res.actual, res.predicted[backend][i]))
            res.metrics[backend][i][:αs] = αs
            res.metrics[backend][i][:πs] = πs
            res.metrics[backend][i][:απ_thresholds] = απ_thresholds
            res.metrics[backend][i][:auROC] = pyconvert(Float64, skmetrics.auc(αs, πs))
        end
    end
end

function metric_error_rate!(res::ForestResults)
    function ϵmin(actual, predicted, tol=1e-6)
        left, right = 0.0, 1.0
        while right - left > tol
            thresholds = range(left, right, length=20)
            scores = mean.(actual .== (predicted .> t) for t in thresholds)
            tbest = argmax(scores)
            left = thresholds[max(1, tbest - 1)]
            right = thresholds[min(length(thresholds), tbest + 1)]
        end
        mean(actual .!= (predicted .> (left+right)/2))
    end
    for backend in keys(res.predicted)
        for i in 1:length(res.predicted[backend])
            res.metrics[backend][i][:error] = ϵmin(res.actual, res.predicted[backend][i][:vals])
        end
    end
end

# ---------------------
# Plotting
# ---------------------

using Gadfly, ColorSchemes

seaborn = pyimport("seaborn")
seaborn.set_palette("Set2")

function binary_pairploter(data::DataFrame, depvar::Symbol)
    bindata = copy(data)
    data[!, depvar] = data[!, depvar] .== mode(data[!, depvar])
    plot = seaborn.pairplot(pytable(data), hue=string(depvar))
    file -> plot.savefig(file)
end

function curve_plot(res::ForestResults, x::Symbol, y::Symbol, extras...)
    data = DataFrame([:backend => Symbol[], :run => Symbol[], x => Float64[], y => Float64[]])
    for backend in keys(res.metrics) |> collect |> sort
        for i in 1:length(res.metrics[backend])
            for j in 1:length(res.metrics[backend][i][x])
                push!(data, [backend, Symbol("$(backend)_$i"),
                             res.metrics[backend][i][x][j],
                             res.metrics[backend][i][y][j]])
            end
        end
    end
    plot(data, x=x, y=y, color=:backend, group=:run,
         Geom.line, Theme(alphas=[0.7]), extras...)
end
