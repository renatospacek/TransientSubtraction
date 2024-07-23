using Molly
using Random
using LinearAlgebra
using DelimitedFiles

# ==============================================================================
function main(η, M; yratio = 1.0, zratio = 1.0)
    Nx = 10
    t_sim = 3.0
    t_therm = 1.0
    dt = 1e-3
    T = 0.8
    ρ = 0.7
    γ = 1.0
    rc = 2.5

    Lx = Nx/cbrt(ρ); Ly = Lx*yratio; Lz = Lx*zratio
    Nsteps = floor(Int64, t_sim/dt); Ntherm = floor(Int64, t_therm/dt)
    Ny = round(Int64, Nx*yratio); Nz = round(Int64, Nx*zratio); N = Nx*Ny*Nz
    rseed = rand(1:1000000000, M)

    sinus_forcing(y) = sin(2π*y/Ly)
    ff = [zeros(Float64, 3) for n in 1:N]

    boundary = CubicBoundary(Lx, Ly, Lz)
    atoms = [Atom(index = i, ϵ = 1.0, σ = 1.0, mass = 1.0) for i = 1:N]

    # Initial conditions which will be thermalized
    q0 = place_atoms_on_3D_lattice(Nx, Ny, Nz, boundary)
    p0 = [random_velocity(1.0, T, 1.0) for n = 1:N] # mass, T, k_B

    lj_w_cutoff = LennardJones(cutoff = ShiftedForceCutoff(rc), force_units = NoUnits, energy_units = NoUnits, use_neighbors = true)
    nf = CellListMapNeighborFinder(eligible = trues(N,N), unit_cell = boundary, n_steps = 10, dist_cutoff = 1.2*rc)
    simulator = LangevinSplitting(dt = dt, temperature = T, friction = γ, splitting="BAOAB",)

    # Definition and thermalizing run for equilibrium dynamics
    sys0 = System(
        atoms = atoms,
        coords = q0,
        velocities = p0,
        boundary = boundary,
        pairwise_inters = (lj_w_cutoff,),
        neighbor_finder = nf,
        force_units = NoUnits,
        energy_units = NoUnits,
        k = 1.0
    )

    simulate!(sys0, simulator, Ntherm)

    # Mapping for transient dynamics p^η = Φ_η = p^0 + ηF(q^0)
    [ff[n][1] = sinus_forcing(sys0.coords[n][2]) for n in 1:N]      
    pη = sys0.velocities .+ η*ff

    # Add loggers to equilibrium system
    sys0 = System(sys0; loggers = (fourier = GeneralObservableLogger(fourier_response, Float64, 1),))

    # Define transient system
    sysη = System(
        atoms = atoms,
        coords = copy(sys0.coords),
        velocities = pη,
        boundary = boundary,
        pairwise_inters = (lj_w_cutoff,),
        neighbor_finder = nf,
        loggers = (fourier = GeneralObservableLogger(fourier_response, Float64, 1),),
        force_units = NoUnits,
        energy_units = NoUnits,
        k = 1.0
    )

    # Write R1, R2 to file after each trajectory
    open("R1_sv_test_$(η)_$(M).out", "w") do f1
        open("R2_sv_test_$(η)_$(M).out", "w") do f2
            for i in 1:M
                simulate!(sys0, simulator, Nsteps, rng=Random.seed!(rseed[i]))
                simulate!(sysη, simulator, Nsteps, rng=Random.seed!(rseed[i]))

                writedlm(f1, values(sys0.loggers.fourier)')
                writedlm(f2, values(sysη.loggers.fourier)')

                empty!(sys0.loggers.fourier.history)
                empty!(sysη.loggers.fourier.history)
                
                # Update map Φ_η
                [ff[n][1] = sinus_forcing(sys0.coords[n][2]) for n in 1:N]        
                pη = sys0.velocities .+ η*ff

                # Reset transient initial conditions for next trajectory
                sysη.coords .= sys0.coords
                sysη.velocities .= pη
            end
        end
    end
end

# ==============================================================================
function fourier_response(s::System, args...; kwargs...)
    p_x = view(reinterpret(reshape, Float64, s.velocities), 1, :)
    q_y = view(reinterpret(reshape, Float64, s.coords), 2, :)
    Ly = s.boundary.side_lengths[2]
    N = length(s)

    return imag(dot(p_x, exp.(2im*π*q_y/Ly))/N)
end

# ==============================================================================
function place_atoms_on_3D_lattice(Nx::Integer, Ny::Integer, Nz::Integer, boundary)
    (Lx,Ly,Lz) = boundary.side_lengths

    return reshape([SVector(i*Lx/Nx, j*Ly/Ny, k*Lz/Nz) for i = 0:Nx-1, j = 0:Ny-1, k = 0:Nz-1], Nx*Ny*Nz)
end

# ==============================================================================
# main(η, M)
@time main(parse(Float64, ARGS[1]), parse(Int64, ARGS[2]))
