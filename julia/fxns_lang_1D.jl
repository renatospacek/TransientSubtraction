# ===============================================================================
@with_kw mutable struct Params
    Lq::AbstractFloat
    Lp::Float64
    mq::Int64
    mp::Int64
    hq::Float64
    hp::Float64
    q::Array{Float64,1}
    p::Array{Float64,1}
    X::Array{Float64,2}
    Y::Array{Float64,2}
    γ::Float64
    β::Float64
    mu0::Array{Float64,2}
    dv::Float64

    function Params(Lp, mq::Int, mp::Int, γ, β)
        Lq = π
        hq = 2Lq/mq
        hp = 2Lp/(mp-1)
        dv = hq*hp

        q = range(-Lq, Lq - hq + 1e-12, mq)
        p = range(-Lp, Lp, mp)
        X0, Y0 = meshgrid(q, p)
        gibbs = _get_gibbs(β)

        X = X0'
        Y = Y0'

        mu0 = normalize_gibbs(β).(X, Y, β)
        new(Lq, Lp, mq, mp, hq, hp, q, p, X, Y, γ, β, mu0, dv)
    end
end

"Defines method for Gibbs distribution"
function _get_gibbs(β::Float64)
    Z = hcubature(x -> exp(-β*V(x[1]) - β*x[2]^2/2), [-π, -6.0], [π, 6.0])[1]
    # println("Z = $Z")
	return ((x, y) -> exp(-β*V(x) - β*y^2/2)/Z)
end

# ==============================================================================
function V(x)
    return (1. - cos(x))/2.0
end

function dV(x)
   return sin(x)/2.0
end

# ==============================================================================
function get_loglog_slope(x, y)
    logx = log.(x)
    logy = log.(y)
    Πx = logx .- mean(logx)
    Πy = logy .- mean(logy)

    return sum(Πx.*Πy)/sum(Πx.^2)
end

function loglog_sl(x, y, step, stop)
    length(y) <= step && return nothing
    return log(y[stop]/y[stop - step])/log(x[stop]/x[stop - step])
end

function clear_label!(series)
    series[:label] = ""
    return series
end

function normalize_gibbs(β)
    integral_p = sqrt(2π/β)
    integral_q, _ = quadgk(x -> exp(-β*V(x)), -π, π)
    Z = integral_p * integral_q
    # println(Z)
    # μ(q,p,β) = exp(-β * V(q) - β * p^2 / 2)/Z

    return ((q,p,β) -> exp(-β*V(q) - β*p^2/2)/Z)
end

# ==============================================================================
function gibbs_p(p, hp, hq, β)
    ψ = exp.(-β.*p.^2 ./2)
    Z = hp*hp*sum(ψ)

    return ψ/Z
end

function μη(X, Y, hq, hp, β, η, ord)
    if ord == 1
        tmp = exp.(-β*V.(X) - β*Y.^2/2 + η*β*Y)
    elseif ord == 2
        tmp = exp.(-β*V.(X) - β*Y.^2/2 + η*β*Y - (η*β*Y).^2/2)
    else
        error("ord should be 1 or 2")
    end

    Z = hq*hp*sum(tmp)
    # println("Z$ord = $Z")

    return tmp/Z
end

function sample_gibbs(V, β, size)
    samples = zeros(size)
    for i in eachindex(samples)
        g = -π + 2π*rand()
        u = rand()
        while u > exp(-β*V(g))
            g = 2π*rand() - π
            u = rand()
        end
        samples[i] = g
    end
    return samples
end