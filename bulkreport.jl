include("genreport.jl")

using RDatasets

generate_report("Edgar Anderson's Iris Data",
                dataset("datasets", "iris"), :Species)

generate_report("Carbon Dioxide Uptake in Grass Plants",
                select(dataset("datasets", "CO2"), :Type, :Conc, :Uptake), :Type)

generate_report("Potency of Orchard Sprays",
                dataset("datasets", "OrchardSprays"), :Treatment)

generate_report("Reaction Velocity of an Enzymatic Reaction",
                dataset("datasets", "Puromycin"), :State)

generate_report("The Effect of Vitamin C on Tooth Growth in Guinea Pigs",
                dataset("datasets", "ToothGrowth"), :Supp)

generate_report("MASS synthentic classification problem TE",
                dataset("MASS", "synth.te"), :YC)

generate_report("MASS synthentic classification problem TR",
                dataset("MASS", "synth.tr"), :YC)

generate_report("Survival from Malignant Melanoma",
                dataset("MASS", "Melanoma"), :Status)

generate_report("Diabetes in Pima Indian Women",
                dataset("MASS", "Pima.te"), :Type)
