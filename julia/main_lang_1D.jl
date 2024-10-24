using Plots
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Parameters
using Random
using StatsBase
using Cubature
using QuadGK
using Interpolations
using LinearSolve
using Printf
using JLD2
plotlyjs()

include("fxns_lang_1D.jl")
include("stencil_lang_1D.jl")

#=
Code to solve compute the η bias for 1D Langevin by discretizing the generator L with
finite differences to solve the PDE -Lf = R
=#

# ==============================================================================
function compute_and_print_results(vars::Params, L, R, solver_name, fvec; print_table=true)
    residual_norm = norm(L * fvec - R)
    f = reshape(fvec, vars.mq, vars.mp)
    avg = sum(f .* vars.mu0)*vars.hq*vars.hp
    f .-= avg

    if print_table
        println(rpad(solver_name, 15), rpad(@sprintf("%.5e", residual_norm), 20), rpad(@sprintf("%.5e", avg), 20))
    end

    return f, residual_norm, avg
end

function lin_solver_data(vars::Params, L, R; plot=false, print_table=true)
    prob = LinearProblem(L, R)
    solvers = [
        # ("linsol", solve(prob).u, :red)
        ("L\\R", L\R, :green),
        # ("lsqr", lsqr(L, R), :blue),
        # ("GMRES()", solve(prob,IterativeSolversJL_GMRES()).u, :black)
    ]

    if print_table
        println(rpad("Solver", 15), rpad("Residual Norm", 20), rpad("Average", 20))
        println("-"^55)
    end
    fs = []

    for (solver_name, sol,) in solvers
        f, res_norm, avg = compute_and_print_results(vars, L, R, solver_name, sol; print_table=print_table)
        push!(fs, f)
    end

    if plot
        s1 = surface(camera=(60,30), size=(800,800))
        scatter!(s1, inset=(1, bbox(0.05,0.1,0.15,0.15)), framestyle=:none, lims=(0,1), fg_color_legend=nothing, legend=:left)

        for i in eachindex(fs)
            color = solvers[i][3]; label = solvers[i][1]
            surface!(s1, vars.X, vars.Y, fs[i], c=cgrad([color, color]), fa=0.8, cb=false)
            scatter!(s1, [0], [-1], msw=0, label=label, subplot=2, color=color)
        end
        display(s1)
    end

    return fs
end

# ==============================================================================
function get_solution()
    mq = 200
    mp = 400
    Lp = 5.0
    vars = Params(Lp, mq, mp, 1.0, 1.0)

    L = getStencil(vars)

    # p = vcat(vars.Y...)
    # R = p.^2 .- p .- 1.0#.- p .- 1.0# .- 1.0# .- 1.0# p.^(1)#.^2

    q = vcat(vars.X...)
    # a = 10.0; b = 10.0; c = 100.0
    # R = (a*cos.(q) .+ b*sin.(q) .+ c*sin.(2q)).*exp.(V.(q))
    a = 1.0; b = -1.0
    R = (a*cos.(q) .+ b*sin.(q)).*exp.(V.(q))

    fs = lin_solver_data(vars, L, R; plot=true)

    return fs, vars
end

function plot_results(vars::Params, fs)
    f = fs[1]
    println(size(fs))
    println(size(f))
    # ηv = [0.2:0.01:0.4;]
    ηv = [0.1:0.025:0.4;]
    # ηv = [0.1]

    int_Phi = zeros(2, length(ηv))
    err = zeros(2, length(ηv))
    ints = zeros(4)

    ints[1] = sum(f.*vars.mu0)*vars.dv
    ints[2] = sum(f'.*vars.p.*vars.mu0')*vars.dv
    ints[3] = sum(f'.*(vars.p.^2 .- 1.0).*vars.mu0')*vars.dv/2
    ints[4] = sum(f'.*(vars.p.^3 .- 3*vars.p).*vars.mu0')vars.dv/6

    Φ1(p, η) = p + η
    Φ2(p, η) = p + η - η^2*p/2

    println("\n\n", rpad("∫f:", 20), @sprintf("%.5g", ints[1]))
    println(rpad("∫fS:", 20), @sprintf("%.5g", ints[2]))
    println(rpad("f(p²-1):", 20), @sprintf("%.5g", ints[3]))
    println(rpad("f(p³-3p):", 20), @sprintf("%.5g", ints[4]))
    println("\n")

    for (j, η) in enumerate(ηv)
        itp = LinearInterpolation((vars.q, vars.p), f, extrapolation_bc = Line())

        fΦ1 = zeros(size(f))
        fΦ2 = zeros(size(f))

        for (j, qj) in enumerate(vars.q), (k, pk) in enumerate(vars.p)
            fΦ1[j, k] = itp(qj, Φ1(pk, η))
            fΦ2[j, k] = itp(qj, Φ2(pk, η))
        end

        int_Phi[1,j] = sum(fΦ1.*vars.mu0)*vars.dv
        int_Phi[2,j] = sum(fΦ2.*vars.mu0)*vars.dv

        err[1,j] = abs.(int_Phi[1,j] - ints[1] - η*ints[2])# - η^2*ints[3]) - η^3*ints[4])
        err[2,j] = abs.(int_Phi[2,j] - ints[1] - η*ints[2])# - η^2*ints[3]) - η^3*ints[4])
    end

    for i in 1:2
    idxs = i:length(ηv)
    pfit1 = polyfit(log10.(ηv[idxs]), log10.(err[1,idxs]), 1)[2];
    pfit2 = polyfit(log10.(ηv[idxs]), log10.(err[2,idxs]), 1)[2];
    println("pfit1 = ", pfit1, " err = ", abs(pfit1-2))
    println("pfit2 = ", pfit2, " err = ", abs(pfit2-3))
    end

    C = [err[1,end]/ηv[end]^2, err[2,end]/ηv[end]^3]
    plt1 = plot()
    plt2 = plot()
    for i in 1:2
        # sl = log(err[i,end]/err[i,end-5])/log(ηv[end]/ηv[end-5])
        sl = loglog_sl(ηv, err[i,:], 3, length(ηv))
        println("Φ^$i Slope = $sl")

        plot!(plt1, ηv, err[i,:], marker=true, label="Φ^$i")
        plot!(plt2, ηv, err[i,:], marker=true, label="", scale=:log10)
    end
    plot!(plt2, t -> C[1]*t^2, label="ref 2", c=:black, ls=:dot, lw=2)
    plot!(plt2, t -> C[2]*t^3, label="ref 3", c=:black, ls=:dash, lw=2)

    plt = plot(plt1, plt2, layout=(2,1), legend=:outertopright, size=(750, 500))
    display(plt)

    # jldsave("1d-lang_bias.jld2"; ηv, err)
end

# ==============================================================================
function tune_observable()
    mq = 100
    mp = 200
    Lp = 5.0
    vars = Params(Lp, mq, mp, 1.0, 1.0)

    L = getStencil(vars)

    # p = vcat(vars.Y...)
    # R = p.^2 .- p .- 1.0#.- p .- 1.0# .- 1.0# .- 1.0# p.^(1)#.^2

    q = vcat(vars.X...)
    # a = 5.0; b = 30.0
    # R = (a*cos.(q) .+ b*sin.(q)).*exp.(V.(q))
    a = 10.0; b = 10.0; c = 100.0; d = 1.0
    R = (a*cos.(q) .+ b*sin.(q) .+ c*sin.(2q) .+ 0*cos.(2q)).*exp.(V.(q))

    fs = lin_solver_data(vars, L, R; plot=false, print_table=false)

    ints_ref = let
        a0 = 1.0; b0 = 1.0; c0 = 1.0; d = 1.0
        R0 = (a0*cos.(q) .+ b0*sin.(q) .+ c0*sin.(2q) .+ 0*cos.(2q)).*exp.(V.(q))
        fs0 = lin_solver_data(vars, L, R0; plot=false, print_table=false)
        f0 = fs0[1]

        [sum(f0.*vars.mu0)*vars.dv,
                sum(f0'.*vars.p.*vars.mu0')*vars.dv,
                sum(f0'.*(vars.p.^2 .- 1.0).*vars.mu0')*vars.dv/2,
                sum(f0'.*(vars.p.^3 .- 3*vars.p).*vars.mu0')vars.dv/6]
    end

    f = fs[1]
    ηv = [0.1:0.05:0.4;]
    # ηv = [0.1]

    int_Phi = zeros(2, length(ηv))
    err = zeros(2, length(ηv))
    ints = zeros(4)

    ints[1] = sum(f.*vars.mu0)*vars.dv
    ints[2] = sum(f'.*vars.p.*vars.mu0')*vars.dv
    ints[3] = sum(f'.*(vars.p.^2 .- 1.0).*vars.mu0')*vars.dv/2
    ints[4] = sum(f'.*(vars.p.^3 .- 3*vars.p).*vars.mu0')vars.dv/6

    Φ1(p, η) = p + η
    Φ2(p, η) = p + η - η^2*p/2

    # println("_"^55)
    # println("\n", rpad("∫f:", 10), @sprintf("%.5g", ints[1]))
    # println(rpad("∫fS:", 10), @sprintf("%.5g", ints[2]))
    # println(rpad("f(p²-1):", 10), @sprintf("%.5g", ints[3]))
    # println(rpad("f(p³-3p):", 10), @sprintf("%.5g", ints[4]))
    # println("\n")

    line_width = 40
    str = "a = $a, b = $b, c = $c"
    padding = max(0, (line_width - length(str)) ÷ 2)
    # Use @sprintf to format the string with padding
    formatted_string = @sprintf("%*s%s%*s", padding, "", str, padding, "")

    println("-"^40)
    println(rpad(formatted_string, line_width))
    println("-"^40)
    println(rpad("Term", 10), rpad("Value", 15), rpad("Ratio", 10))
    println(rpad("∫f:", 10), rpad(@sprintf("%.5g", ints[1]), 15), @sprintf("%.5g", ints[1]/ints_ref[1]))
    println(rpad("∫fS:", 10), rpad(@sprintf("%.5g", ints[2]), 15), @sprintf("%.5g", ints[2]/ints_ref[2]))
    println(rpad("f(p²-1):", 10), rpad(@sprintf("%.5g", ints[3]), 15), @sprintf("%.5g", ints[3]/ints_ref[3]))
    println(rpad("f(p³-3p):", 10), rpad(@sprintf("%.5g", ints[4]), 15), @sprintf("%.5g", ints[4]/ints_ref[4]))
    println("-"^40)
    # println("\n")
end

# ==============================================================================
# %%
@time fs, vars = get_solution();

# # %%
@time plot_results(vars, fs)
# @time tune_observable()