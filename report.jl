using Org, Gadfly
using Dates

const REPORT_DIR = joinpath(@__DIR__, "reports")
const FIGURE_DIR = "figures"

if !isdir(REPORT_DIR)
    mkdir(REPORT_DIR)
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
end

function Base.convert(::Type{Org.Component}, pg::PlotGrid)
    fracwidth = round(0.98 / length(pg.plots), digits=3)
    Org.SpecialBlock(
        "center",
        map(pg.plots) do wp
            Org.AffiliatedKeywordsWrapper(
                convert(Org.Component, wp),
                ["attr_latex" => ":width $fracwidth\\linewidth :center"])
        end)
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
                                        string(Org.TimestampInactive(now()))[2:end-1])])]))

exportcmd(path::String, exporter::String) =
    `emacs --batch --eval
"(progn (load \"~/.emacs.d/init.el\") (require 'org) (require 'ox-latex) (setq org-latex-default-figure-position \"!htbp\"))"
$path
-f "$exporter"`

function Base.write(r::Report, type::Symbol=:org)
    orgfile = joinpath(REPORT_DIR, replace(r.name, r"[^A-Za-z0-9]+" => "-") * ".org")
    if !r.orgwritten || type == :rmorg
        write(orgfile, sprint(org, convert(OrgDoc, r)))
    end
    if type == :pdf
        run(Cmd(exportcmd(orgfile, "org-latex-export-to-pdf"), dir=REPORT_DIR), wait=false)
    elseif type == :html
        run(Cmd(exportcmd(orgfile, "org-html-export-to-html"), dir=REPORT_DIR), wait=false)
    elseif type == :rmorg && isfile(orgfile)
        rm(orgfile)
    end
end
