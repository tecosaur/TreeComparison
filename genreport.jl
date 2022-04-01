include("evaluation.jl")
include("report.jl")

function generate_report(name::String, data::DataFrame, depvar::Symbol,
                         config::ForestConfig{Vector{Int}}=ForestConfig(ntrees=[50, 100, 200, 500, 1000]),
                         backends::Vector{Symbol}=forest_backends)
    start = time()
    # Report
    printstyled("Generating report on $name\n", color=:blue)
    report = Report(name)
    depvarmode = mode(data[!, depvar])
    push!(report,
          Org.Section([
              Org.Keyword("macro" =>
                  "color " *
                  string(Org.Paragraph([
                      Org.ExportSnippet("latex", "\\textcolor[HTML]{\$1}{"),
                      Org.ExportSnippet("html", "<span style=\"color: #\$1\">"),
                      Org.TextPlain("\$2"),
                      Org.ExportSnippet("latex", "}"),
                      Org.ExportSnippet("html", "</span>")]))),
              parse(Org.Paragraph,
                string("The $name ",
                       join(size(data), "\\times"),
                       " dataset has ",
                       length(unique(data[!, depvar])),
                       " $depvar classes, the most common of which, /",
                       mode(data[!, depvar]),
                       "/ comprises ",
                       round(Int, 100 * mean(data[!, depvar] .== depvarmode)),
                       "% of the data. This is used to form a two-class dataset.")),
          parse(Org.Paragraph,
                string("The $name data is classified with ",
                       length(backends),
                       " different random forrest implementations: ",
                       join(string.('~', backends, '~'), ", ", ", and "), ". ",
                       "In each instance, ",
                       join(string.(config.ntrees), ", ", ", and "),
                       " trees are used, checking ",
                       Dict(:all => "all $(size(data, 2))",
                            :sqrt => "\\(\\lfloor \\sqrt{$(size(data, 2))} \\rfloor = $(FC_mtry[config.mtry](size(data, 2)))\\)",
                            :log2 => "\\(\\lfloor \\log_2 $(size(data, 2)) \\rfloor = $(FC_mtry[config.mtry](size(data, 2)))\\)",
                            :third => "\\(\\lfloor $(size(data, 2))/3 \\rfloor = $(FC_mtry[config.mtry](size(data, 2)))\\)",
                            )[config.mtry],
                       " variables each split, ",
                       "and the out-of-bag class probabilities analysed.")),
            Org.Keyword("latex" => "\\newpage")]))
    push!(report, Org.Heading(1, "Data visualisation", Org.Section()))
    push!(report, Plotter(binary_pairploter(data, depvar),
                          ["attr_latex" => ":width 0.6\\linewidth"]))
    push!(report, Org.Keyword("latex" => "\\newpage"))
    # Note spot for the overall summary
    overallsummarypos = length(report.content)
    scalar_metrics = [("auPRC", :auPRC), ("auROC", :auROC), ("maximum F1 score", :f1max)]
    scalar_metric_results = DataFrame(ntrees=Int[], backend=Symbol[] ;NamedTuple(zip(last.(scalar_metrics), fill(Float64[], length(scalar_metrics))))...)
    consistency_scores = DataFrame(ntrees=Int[], ;NamedTuple(zip(last.(scalar_metrics), fill(Float64[], length(scalar_metrics))))...)
    # Per-ntree results
    for fconfig in config
        push!(report, Org.Keyword("latex" => "\\newpage"))
        treeres = Org.Heading(1, "$(fconfig.ntrees) trees", Org.Section())
        # Generate results
        printstyled("Generating $(fconfig.ntrees) tree results\n", color=:blue)
        results = forest_preds(data, depvar, fconfig, backends)
        metric_precrec!(results)
        metric_roc!(results)
        for backend in keys(results.metrics) |> collect |> sort
            for i in 1:length(results.metrics[backend])
                push!(scalar_metric_results,
                      vcat(fconfig.ntrees, backend,
                           getindex.(Ref(results.metrics[backend][i]),
                                     last.(scalar_metrics))))
            end
        end
        # Summary
        consistencies = Float64[]
        summary = Org.Table(vcat(
            Org.TableRow(vcat("Metric", string.(backends), "consistency")),
            Org.TableHrule(),
            map(scalar_metrics) do (label, prop)
                medscores = map(backends) do backend
                    getindex.(results.metrics[backend], prop) |> median
                end |> m -> replace(m, NaN => 0)
                consistency = 1 - sqrt(2var(medscores / maximum(medscores)))
                push!(consistencies, consistency)
                Org.TableRow(
                    vcat(label,
                        string.(round.(medscores, digits=4)),
                        if isnan(var(medscores)) "1"
                        else string(round(consistency, digits=2))
                        end))
            end))
        push!(consistency_scores, vcat(fconfig.ntrees, consistencies))
        push!(treeres.section.contents,
            Org.AffiliatedKeywordsWrapper(
                summary,
                ["caption" => "Median metrics for each implementation.",
                 "attr_latex" => ":align l|" * 'l'^length(backends) * "|l"]))
        push!(report, treeres)
        # Curves
        push!(report, org"Precision-recall and ROC curves."p)
        push!(report,
              PlotGrid([Plotter(curve_plot(results, :recall, :precision,
                                           Coord.cartesian(xmin=0, xmax=1, ymin=0, ymax=1))),
                        Plotter(curve_plot(results, :αs, :πs,
                                           Coord.cartesian(xmin=0, xmax=1, ymin=0, ymax=1)))]))
        # Density
        scalar_densities = DataFrame(backend=Symbol[], ;NamedTuple(zip(last.(scalar_metrics), fill(Float64[], length(scalar_metrics))))...)
        for backend in keys(results.metrics) |> collect |> sort
            for i in 1:length(results.metrics[backend])
                push!(scalar_densities,
                      vcat(backend,
                           replace(getindex.(Ref(results.metrics[backend][i]), last.(scalar_metrics)),
                                   NaN => 0.0)))
            end
        end
        push!(report, org"Distribution of scalar scores."p)
        push!(report, PlotGrid(
            Plotter.(plot(scalar_densities,
                          y=metric, x=:backend, color=:backend,
                          Geom.boxplot,
                          Coord.cartesian(ymin=0, ymax=1))
                     for metric in last.(scalar_metrics))))
        push!(report, Org.Keyword("latex", "\\newpage"))
        for (label, prop) in scalar_metrics
            push!(report, Org.AffiliatedKeywordsWrapper(
                Org.Table(vcat(
                    Org.TableRow(vcat("Score", string.(1:fconfig.replicates))),
                    Org.TableHrule(),
                    map(keys(results.metrics) |> collect |> sort) do backend
                        scores = getindex.(results.metrics[backend], prop)
                        Org.TableRow(string.(vcat(backend, round.(scores, digits=3))))
                    end)),
                ["caption" => "Raw $label scores by backend.",
                 "attr_latex" => ":align l|" * 'l'^fconfig.replicates]))
        end
    end
    # Overall results
    insert!(report.content, (overallsummarypos+=1),
            Org.Heading(1, "Forrest performance with increasing trees"))

    insert!(report.content, (overallsummarypos+=1),
            Org.Table(vcat(
                Org.TableRow(vcat("n(trees)", string.(names(consistency_scores)[2:end], " consistency"))),
                Org.TableHrule(),
                Org.TableRow.(collect.(string.(round.(vals, digits=3)) for vals in
                                           values.(eachrow(consistency_scores)))))))

    insert!(report.content, (overallsummarypos+=1),
            org"Consistency is calculated as one take the normalised standard deviation
                --- i.e. \(1-\sqrt{2} \cdot \sigma(x / \max(x))\)."p)

    insert!(report.content, (overallsummarypos+=1),
            PlotGrid(map(last.(scalar_metrics)) do metric
                         Plotter(plot(scalar_metric_results,
                                      x=:ntrees, y=metric, color=:backend,
                                      group=:backend, Geom.point, size=[5pt],
                                      Geom.smooth,
                                      Theme(alphas=[0.3], discrete_highlight_color=c->nothing),
                                      Coord.cartesian(ymin=0, ymax=1)))
                     end))
    # Finish up
    printstyled("Saving report\n", color=:blue)
    write(report, :pdf)
    write(report, :html)
    printstyled("Finished in ", round(Int, time() - start), "s\n", color=:blue)
end
