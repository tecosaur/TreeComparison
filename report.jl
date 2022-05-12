using Org, Gadfly
using Dates

const REPORT_DIR = joinpath(@__DIR__, "reports")
const FIGURE_DIR = "figures"

if !isdir(REPORT_DIR)
    mkdir(REPORT_DIR)
end
if !isdir(joinpath(REPORT_DIR, FIGURE_DIR))
    mkdir(joinpath(REPORT_DIR, FIGURE_DIR))
end

struct Plotter
    title::Union{Org.Paragraph, Nothing}
    plotter::Function # filename::String -> Nothing
    attributes::Vector{Pair}
end
Plotter(plotter::Function, attributes::Vector{<:Pair}=Pair[]) = Plotter(nothing, plotter, attributes)

Plotter(title::Union{Org.Paragraph, Nothing}, plot::Plot, attributes::Vector{<:Pair}=Pair[], drawargs::NamedTuple=(;)) =
    Plotter(title, file -> draw(SVG(file, drawargs...), plot), attributes)
Plotter(plot::Plot, attributes::Vector{<:Pair}=Pair[], drawargs::NamedTuple=(;)) =
    Plotter(nothing, plot, attributes, drawargs)

function Base.convert(::Type{Org.Component}, pltr::Plotter)
    img = string(if !isnothing(pltr.title)
                     replace(replace(string(pltr.title), r"[*/~=_]" => ""),
                             r"[^A-Za-z0-9]+" => "-")
                 else "fig" end,
                 '-', string(rand(UInt32), base=36),
                 ".svg")
    pltr.plotter(joinpath(REPORT_DIR, FIGURE_DIR, img))
    Org.Paragraph(Org.RegularLink("file", joinpath(FIGURE_DIR, img)))
end

struct PlotGrid
    plots::Vector{Plotter}
    nrow::Integer
end

PlotGrid(plots::Vector{Plotter}) = PlotGrid(plots, 1)

function Base.convert(::Type{Org.Component}, pg::PlotGrid)
    ncol = floor(length(pg.plots) / pg.nrow)
    fracwidth = round(0.98 / ncol, digits=3)
    plotblock(plots::Vector{Plotter}) =
        Org.SpecialBlock(
            "center",
            map(plots) do p
                Org.AffiliatedKeywordsWrapper(
                    convert(Org.Component, p),
                    ["attr_latex" => ":width $fracwidth\\linewidth :center"])
            end)
    if nrow == 1
        plotblock(pg.plots)
    else
        Org.Drawer("plotgrid",
                   [plotblock(pg.plots[UnitRange{Int64}(ps, min(length(pg.plots), ps+ncol))])
                    for ps in
                    1:ncol:length(pg.plots)-(ncol-1)])
    end
end

mutable struct Report
    name::String
    content::Vector
    orgwritten::Bool
end

Report(name::String, content::Vector) = Report(name, Vector{Any}(content), false)
Report(name::String) = Report(name, Any[])

Base.push!(r::Report, x) = push!(r.content, x)

Base.convert(::Type{OrgDoc}, r::Report) =
    OrgDoc(reduce(
        function (secs, next)
            if next isa Org.Heading
                push!(secs, deepcopy(next))
            elseif secs[end] isa Org.Section
                secs[end] *= next
            elseif isnothing(secs[end].section)
                secs[end].section = Org.Section() * next
            else
                secs[end].section *= next
            end
            secs
        end,
        map(r.content) do c
            if c isa Org.Component; c
            else convert(Org.Component, c) end
        end,
        init=Union{Org.Heading, Org.Section}[
            Org.Section(Org.Element[Org.Keyword("title" => r.name),
                                    Org.Keyword("date" =>
                                        string(Org.TimestampInactive(now()))[2:end-1]),
                                    Org.Keyword("latex_header", "\\usepackage[top=2.5cm,bottom=3cm]{geometry}"),
                                    Org.Keyword("latex_header", "\\usepackage[inkscapelatex=false]{svg}"),
                                    Org.Keyword("options", "coverpage:no")])]))

exportcmd(path::String, exporter::String) =
    `emacs --batch --eval
"(progn (when (file-exists-p \"~/.emacs.d/init.el\") (load \"~/.emacs.d/init.el\")) (require 'org) (require 'ox-latex) (setq org-latex-default-figure-position \"!htbp\"))"
$path
-f "$exporter"`

function Base.write(r::Report, type::Symbol=:org)
    orgfile = joinpath(REPORT_DIR, replace(r.name, r"[^A-Za-z0-9]+" => "-") * ".org")
    if !r.orgwritten || type == :rmorg
        write(orgfile, sprint(org, convert(OrgDoc, r)))
    end
    if type == :pdf
            if any(!isnothing, (Sys.which("latexmk"), Sys.which("pdflatex")))
                run(Cmd(exportcmd(orgfile, "org-latex-export-to-pdf"), dir=REPORT_DIR), wait=false)
            else
                run(Cmd(exportcmd(orgfile, "org-latex-export-to-latex"), dir=REPORT_DIR), wait=false)
            end
    elseif type == :html
        run(Cmd(exportcmd(orgfile, "org-html-export-to-html"), dir=REPORT_DIR), wait=false)
    elseif type == :rmorg && isfile(orgfile)
        rm(orgfile)
    end
end
