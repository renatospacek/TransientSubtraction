using JLD2
# using Plots
# plotlyjs()

# Option 1 ==============================
abstract type TransientSubtractionData end

mutable struct Mobility <: TransientSubtractionData
    R1::Array{Float64,2}
    R2::Array{Float64,2}
    dR::Array{Float64,2}
    cR1::Array{Float64,2}
    cR2::Array{Float64,2}
    cdR::Array{Float64,2}
    cv1::Array{Float64,2}
    cv2::Array{Float64,2}
    cvd::Array{Float64,2}
    ηv::Array{Float64,1}
    M::Int
    Nsteps::Int
    dt::Float64
    tv::Array{Float64,1}

    function Mobility(filename::String)
        instance = new()
        initialize_from_file!(instance, filename)
        return instance
    end
end
    
function initialize_from_file!(instance::TransientSubtractionData, filename::String)
    jldopen(filename, "r") do f
        fields = fieldnames(typeof(instance))
        map(field -> setproperty!(instance, field, f[string(field)]), fields)
    end
end

# Option 2 ==============================
mutable struct Mobility
    R1::Array{Float64,2}
    R2::Array{Float64,2}
    dR::Array{Float64,2}
    cR1::Array{Float64,2}
    cR2::Array{Float64,2}
    cdR::Array{Float64,2}
    cv1::Array{Float64,2}
    cv2::Array{Float64,2}
    cvd::Array{Float64,2}
    ηv::Array{Float64,1}
    M::Int
    Nsteps::Int
    dt::Float64
    tv::Array{Float64,1}

    function Mobility(filename::String)
        instance = new()

        jldopen(filename, "r") do f
            fields = fieldnames(Mobility)
            map(field -> setproperty!(instance, field, f[string(field)]), fields)
        end

        return instance
    end
end

function mobility_new()
    # f = load("sims_data/upt_mobility_data.jld2")
    aaa = Mobility("sims_data/upt_mobility_data.jld2")

    println("done")
   
    return aaa
end

@time aaa=mobility_new()
