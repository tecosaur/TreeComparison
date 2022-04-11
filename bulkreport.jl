include("genreport.jl")

using RDatasets

cases = [
    ("Iris", "Edgar Anderson's Iris Data",
     dataset("datasets", "iris"), :Species),
    ("CO2", "Carbon Dioxide Uptake in Grass Plants",
     select(dataset("datasets", "CO2"), :Type, :Conc, :Uptake), :Type),
    ("Orchard", "Potency of Orchard Sprays",
     dataset("datasets", "OrchardSprays"), :Treatment),
    ("EnzRVol", "Reaction Velocity of an Enzymatic Reaction",
     dataset("datasets", "Puromycin"), :State),
    ("Tooth", "The Effect of Vitamin C on Tooth Growth in Guinea Pigs",
     dataset("datasets", "ToothGrowth"), :Supp),
    ("Synth.TE", "MASS synthentic classification problem TE",
     dataset("MASS", "synth.te"), :YC),
    ("Synth.TR", "MASS synthentic classification problem TR",
     dataset("MASS", "synth.tr"), :YC),
    ("Melanoma", "Survival from Malignant Melanoma",
     dataset("MASS", "Melanoma"), :Status),
    ("Pima", "Diabetes in Pima Indian Women",
     dataset("MASS", "Pima.te"), :Type)
]

overall_results =
    DataFrame(map(cases) do (nameshort, name, data, depvar)
                  consistency_scores, scalar_scores = generate_report(name, data, depvar)
                  merge((;data=nameshort), NamedTuple(consistency_scores[end, :]))
              end)
select!(overall_results, Not(:ntrees))

overall_summary =
    plot(stack(overall_results, Not(:data),
               variable_name=:metric, value_name=:consistency),
         x=:data, y=:consistency, color=:metric, Geom.line)

draw(SVG(joinpath(REPORT_DIR, "consistency.svg")), overall_summary)
