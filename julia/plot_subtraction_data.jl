using JLD2
using Plots
plotlyjs()

# TODOs:
# - also add to jld2 file (or python data file): tv
# - 

# ===============================================================================
function main(sys)
    f = load("$(sys)_data.jld2")
    extract(f)

    tv = [1:Nsteps;]*dt
    ρref = sys == "sv" ? 0.322 : 0.121

    cR1 = dt*cumsum(R1, dims=2)./ηv
    cR2 = dt*cumsum(R2, dims=2)./ηv
    cdR = dt*cumsum(dR, dims=2)./ηv

    ind = 1
    plt1 = plot(yaxis="Instantaneous response", title="LJ $sys (η = $(ηv[ind]), M = $M)")
    plot!(plt1, tv, R1[ind,:], label="η = 0.0")
    plot!(plt1, tv, R2[ind,:], label="η = $(ηv[ind])")
    plot!(plt1, tv, dR[ind,:], label="Subtr.")

    plt2 = plot(yaxis="ρhat (integrated res.)/η")
    plot!(plt2, tv, cR1[ind,:], label="", ribbon=sqrt.(v1[ind,:]/M), fillalpha=0.5)
    plot!(plt2, tv, cR2[ind,:], label="", ribbon=sqrt.(v2[ind,:]/M), fillalpha=0.5)
    plot!(plt2, tv, cdR[ind,:], label="", ribbon=sqrt.(vd[ind,:]/M), fillalpha=0.5)
    plot!(plt2, tv -> ρref, color=:black, linestyle=:dash, label="Ref. value")

    plt3 = plot(ylabel="Variance", xlabel="time")
    plot!(plt3, tv, v2[ind,:], label="", color=2)
    plot!(plt3, tv, vd[ind,:], label="", color=3)

    plt4 = plot(yaxis="error |ρ - ρhat|")
    plot!(plt4, tv, abs.(cdR[ind,:] .- ρref), label="", color=3)

    pltF = plot(plt1, plt2, layout=(2,1), size=(750,500), legend=:outertopright)
    pltF2 = plot(plt1, plt2, plt3, layout=(3,1), size=(750,750), legend=:outertopright)
    pltF3 = plot(plt1, plt2, plt3, plt4, layout=(4,1), size=(750,900), legend=:outertopright)

    display(pltF2)
end

# ===============================================================================
function extract(d)
    expr = quote end
    for (k, v) in d
        push!(expr.args, :($(Symbol(k)) = $v))
    end
    eval(expr)
    return
end

# ===============================================================================
# @time main("mobility")
@time main("sv")
