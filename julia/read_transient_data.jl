using Random
using LinearAlgebra
using DelimitedFiles
using StatsBase
using JLD2

# ===============================================================================
function read_data_mobility()
    ηv = [0.01, 0.1, 1.0]
    M = 20000
    Nsteps = 2001
    dt = 1e-3

    R1 = zeros(length(ηv), Nsteps)
    R2 = zeros(length(ηv), Nsteps)
    dR = zeros(length(ηv), Nsteps)

    v1 = zeros(length(ηv), Nsteps)
    v2 = zeros(length(ηv), Nsteps)
    vd = zeros(length(ηv), Nsteps)

    for i in eachindex(ηv)
        R1tmp = readdlm("R1_mobility_test_$(ηv[i])_$M.out")
        R2tmp = readdlm("R2_mobility_test_$(ηv[i])_$M.out")
        dRtmp = R2tmp - R1tmp

        R1[i,:] = vec(mean(R1tmp, dims=1))
        R2[i,:] = vec(mean(R2tmp, dims=1))
        dR[i,:] = vec(mean(dRtmp, dims=1))

        v1[i,:] = vec(var(R1tmp, dims=1))
        v2[i,:] = vec(var(R2tmp, dims=1))
        vd[i,:] = vec(var(dRtmp, dims=1))
    end

    jldsave("mobility_data.jld2"; R1, R2, dR, v1, v2, vd, ηv, M, Nsteps, dt)
end

# ===============================================================================
function read_data_sv()
    ηv = [0.01, 0.1]
    M = 100000
    Nsteps = 3001
    dt = 1e-3

    R1 = zeros(length(ηv), Nsteps)
    R2 = zeros(length(ηv), Nsteps)
    dR = zeros(length(ηv), Nsteps)

    v1 = zeros(length(ηv), Nsteps)
    v2 = zeros(length(ηv), Nsteps)
    vd = zeros(length(ηv), Nsteps)

    for i in eachindex(ηv)
        R1tmp = readdlm("R1_sv_test_$(ηv[i])_$M.out")
        R2tmp = readdlm("R2_sv_test_$(ηv[i])_$M.out")
        dRtmp = R2tmp - R1tmp

        R1[i,:] = vec(mean(R1tmp, dims=1))
        R2[i,:] = vec(mean(R2tmp, dims=1))
        dR[i,:] = vec(mean(dRtmp, dims=1))

        v1[i,:] = vec(var(R1tmp, dims=1))
        v2[i,:] = vec(var(R2tmp, dims=1))
        vd[i,:] = vec(var(dRtmp, dims=1))
    end

    jldsave("sv_data.jld2"; R1, R2, dR, v1, v2, vd, ηv, M, Nsteps, dt)
end

# ===============================================================================
@time read_data_mobility()
@time read_data_sv()
