# ==============================================================================
aL0(q, p, hq, hp, γ, β) = -p/2/hq
bL0(q, p, hq, hp, γ, β) = p/2/hp*γ + 1/hp^2*γ/β + dV(q)/2/hp
cL0(q, p, hq, hp, γ, β) = -2/hp^2*γ/β
dL0(q, p, hq, hp, γ, β) = -p/2/hp*γ + 1/hp^2*γ/β - dV(q)/2/hp
eL0(q, p, hq, hp, γ, β) = p/2/hq

# ==============================================================================
# Discretizes L
function getStencil(vars::Params)
    @unpack_Params vars

    iV = Int[]
    jV = Int[]
    valV = Float64[]

    # main diagonal
    l = 1
    for i in 1:mp, j in 1:mq
        push!(iV,l)
        push!(jV,l)
        push!(valV, cL0(q[j], p[i], hq, hp, γ, β))
        l += 1
    end

    # upper and lower
    for i in 1:mp, j in 1:mq-1
        push!(iV,(i-1)*mq+j)
        push!(jV,(i-1)*mq+j+1)
        push!(valV, eL0(q[j+1], p[i], hq, hp, γ, β))

        push!(jV,(i-1)*mq+j)
        push!(iV,(i-1)*mq+j+1)
        push!(valV, aL0(q[j], p[i], hq, hp, γ, β))
    end

    # other upper and lower
    for i in 2:mp, j in 1:mq
        push!(iV,(i-2)*mq+j)
        push!(jV,(i-1)*mq+j)
        push!(valV, dL0(q[j], p[i-1], hq, hp, γ, β))

        push!(jV,(i-2)*mq+j)
        push!(iV,(i-1)*mq+j)
        push!(valV, bL0(q[j], p[i], hq, hp, γ, β))
    end

    # PBCs for q
    for i in 1:mp
        push!(iV,(i-1)*mq+1)
        push!(jV,i*mq)
        push!(valV, aL0(q[end], p[i], hq, hp, γ, β))

        push!(jV,(i-1)*mq+1)
        push!(iV,i*mq)
        push!(valV, eL0(q[1], p[i], hq, hp, γ, β))
    end

    return sparse(iV, jV, valV)
end