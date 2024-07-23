using Molly
using Random
using LinearAlgebra
using DelimitedFiles

# ==============================================================================
function main(η, M)
    Npart = 10
    t_sim = 2.0
    t_therm = 1.0
    dt = 1e-3
    T = 1.25
    ρ = 0.6
    γ = 1.0
    rc = 2.5

    L = Npart/cbrt(ρ)
    Nsteps = floor(Int64, t_sim/dt); Ntherm = floor(Int64, t_therm/dt)
    N = Npart^3
    rseed = rand(1:1000000000, M)

    boundary = CubicBoundary(L, L, L)
    atoms = [Atom(index = i, ϵ = 1.0, σ = 1.0, mass = 1.0) for i = 1:N]

    # Initial conditions which will be thermalized
    q0 = place_atoms_on_3D_lattice(Npart, L)
    p0 = [random_velocity(1.0, T, 1.0) for n = 1:N] # mass, T, k_B

    lj_w_cutoff = LennardJones(cutoff = ShiftedForceCutoff(rc), force_units = NoUnits, energy_units = NoUnits, use_neighbors = true)
    nf = CellListMapNeighborFinder(eligible = trues(N,N), unit_cell = boundary, n_steps = 10, dist_cutoff = 1.2*rc)
    simulator = LangevinSplitting(dt = dt, temperature = T, friction = γ, splitting="BAOAB",)

    # Define color drift forcing
    cd_forcing = [(-1.0)^i for i in 1:N]./sqrt(N)
    ff = [[cd_forcing[i], 0.0, 0.0] for i in 1:N]
    
    # Definition and thermalizing run for equilibrium dynamics
    sys0 = System(
        atoms = atoms,
        coords = q0,
        velocities = p0,
        boundary = boundary,
        pairwise_inters = (lj_w_cutoff,),
        neighbor_finder = nf,
        loggers = (colordrift = GeneralObservableLogger(colordrift_response, Float64, 1),),
        force_units = NoUnits,
        energy_units = NoUnits,
        k = 1.0,
        data = (forcing = cd_forcing,)
    )

    simulate!(sys0, simulator, Ntherm; run_loggers=false)

    # Mapping for transient dynamics p^η = Φ_η = p^0 + ηF(q^0)
    pη = sys0.velocities .+ η*ff

    # Define transient system
    sysη = System(
        atoms = atoms,
        coords = copy(sys0.coords),
        velocities = pη,
        boundary = boundary,
        pairwise_inters = (lj_w_cutoff,),
        neighbor_finder = nf,
        loggers = (colordrift = GeneralObservableLogger(colordrift_response, Float64, 1),),
        force_units = NoUnits,
        energy_units = NoUnits,
        k = 1.0,
        data = (forcing = cd_forcing,)
    )

    # Write R1, R2 to file after each trajectory
    open("R1_mobility_test_$(η)_$(M).out", "w") do f1
        open("R2_mobility_test_$(η)_$(M).out", "w") do f2
            for i in 1:M
                simulate!(sys0, simulator, Nsteps, rng=Random.seed!(rseed[i]))
                simulate!(sysη, simulator, Nsteps, rng=Random.seed!(rseed[i]))

                writedlm(f1, values(sys0.loggers.colordrift)')
                writedlm(f2, values(sysη.loggers.colordrift)')

                empty!(sys0.loggers.colordrift.history)
                empty!(sysη.loggers.colordrift.history)
                
                # Update map Φ_η
                pη = sys0.velocities .+ η*ff

                # Reset transient initial conditions for next trajectory
                sysη.coords .= sys0.coords
                sysη.velocities .= pη
            end
        end
    end
end

# ==============================================================================
function colordrift_response(s::System, args...; kwargs...)
    p_x = view(reinterpret(reshape, Float64, s.velocities), 1, :)

    return dot(p_x, s.data.forcing)
end

# ==============================================================================
function place_atoms_on_3D_lattice(N::Integer, L)
    reshape([SVector(i*L/N, j*L/N, k*L/N) for i = 0:N-1, j = 0:N-1, k = 0:N-1], N^3)
end

# ==============================================================================
# main(η, M)
@time main(parse(Float64, ARGS[1]), parse(Int64, ARGS[2]))
