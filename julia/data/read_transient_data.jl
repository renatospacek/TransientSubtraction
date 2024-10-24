using Random
using LinearAlgebra
using DelimitedFiles
using StatsBase
using JLD2

# ===============================================================================
function read_data_mobility()
    ηv = [0.01, 0.1, 1.0]
    M = 100000
    Nsteps = 2001
    dt = 1e-3
    tv = [1:Nsteps;]*dt

    R1 = zeros(length(ηv), Nsteps)
    R2 = zeros(length(ηv), Nsteps)
    dR = zeros(length(ηv), Nsteps)

    cR1 = zeros(length(ηv), Nsteps)
    cR2 = zeros(length(ηv), Nsteps)
    cdR = zeros(length(ηv), Nsteps)

    v1 = zeros(length(ηv), Nsteps)
    v2 = zeros(length(ηv), Nsteps)
    vd = zeros(length(ηv), Nsteps)

    cv1 = zeros(length(ηv), Nsteps)
    cv2 = zeros(length(ηv), Nsteps)
    cvd = zeros(length(ηv), Nsteps)

    for i in eachindex(ηv)
        R1tmp = readdlm("R1_mobility_test_$(ηv[i])_$M.out")
        R2tmp = readdlm("R2_mobility_test_$(ηv[i])_$M.out")
        dRtmp = R2tmp - R1tmp

        cR1tmp = dt*cumsum(R1tmp, dims=2)/ηv[i]
        cR2tmp = dt*cumsum(R2tmp, dims=2)/ηv[i]
        cdRtmp = dt*cumsum(dRtmp, dims=2)/ηv[i]

        R1[i,:] = vec(mean(R1tmp, dims=1))
        R2[i,:] = vec(mean(R2tmp, dims=1))
        dR[i,:] = vec(mean(dRtmp, dims=1))

        cR1[i,:] = vec(mean(cR1tmp, dims=1))
        cR2[i,:] = vec(mean(cR2tmp, dims=1))
        cdR[i,:] = vec(mean(cdRtmp, dims=1))

        v1[i,:] = vec(var(R1tmp, dims=1))
        v2[i,:] = vec(var(R2tmp, dims=1))
        vd[i,:] = vec(var(dRtmp, dims=1))

        cv1[i,:] = vec(var(cR1tmp, dims=1))
        cv2[i,:] = vec(var(cR2tmp, dims=1))
        cvd[i,:] = vec(var(cdRtmp, dims=1))
    end

    jldsave("mobility_data.jld2"; R1, R2, dR, cR1, cR2, cdR, v1, v2, vd, cv1, cv2, cvd, ηv, M, Nsteps, dt, tv)
end

# ===============================================================================
function read_data_sv(yratio, zratio)
    ηv = [0.01, 0.1]
    M = 100000
    Nsteps = 3501
    dt = 1e-3
    tv = [1:Nsteps;]*dt

    R1 = zeros(length(ηv), Nsteps)
    R2 = zeros(length(ηv), Nsteps)
    dR = zeros(length(ηv), Nsteps)

    cR1 = zeros(length(ηv), Nsteps)
    cR2 = zeros(length(ηv), Nsteps)
    cdR = zeros(length(ηv), Nsteps)

    v1 = zeros(length(ηv), Nsteps)
    v2 = zeros(length(ηv), Nsteps)
    vd = zeros(length(ηv), Nsteps)

    cv1 = zeros(length(ηv), Nsteps)
    cv2 = zeros(length(ηv), Nsteps)
    cvd = zeros(length(ηv), Nsteps)

    for i in eachindex(ηv)
        R1tmp = readdlm("R1_sv_test_$(ηv[i])_$(M)_$(yratio)_$(zratio).out")
        R2tmp = readdlm("R2_sv_test_$(ηv[i])_$(M)_$(yratio)_$(zratio).out")
        dRtmp = R2tmp - R1tmp

        cR1tmp = dt*cumsum(R1tmp, dims=2)/ηv[i]
        cR2tmp = dt*cumsum(R2tmp, dims=2)/ηv[i]
        cdRtmp = dt*cumsum(dRtmp, dims=2)/ηv[i]

        R1[i,:] = vec(mean(R1tmp, dims=1))
        R2[i,:] = vec(mean(R2tmp, dims=1))
        dR[i,:] = vec(mean(dRtmp, dims=1))

        cR1[i,:] = vec(mean(cR1tmp, dims=1))
        cR2[i,:] = vec(mean(cR2tmp, dims=1))
        cdR[i,:] = vec(mean(cdRtmp, dims=1))

        v1[i,:] = vec(var(R1tmp, dims=1))
        v2[i,:] = vec(var(R2tmp, dims=1))
        vd[i,:] = vec(var(dRtmp, dims=1))

        cv1[i,:] = vec(var(cR1tmp, dims=1))
        cv2[i,:] = vec(var(cR2tmp, dims=1))
        cvd[i,:] = vec(var(cdRtmp, dims=1))
    end

    jldsave("sv_data_$(yratio)_$(zratio).jld2"; R1, R2, dR, cR1, cR2, cdR, v1, v2, vd, cv1, cv2, cvd, ηv, M, Nsteps, dt, tv, yratio, zratio)
end

#@time read_data_mobility()
#@time read_data_sv(yratio, zratio)
