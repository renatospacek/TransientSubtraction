using Plots
using Molly
using Random
using LinearAlgebra
using Statistics
using Printf

#=
Code to visualize the relaxation of the LJ fluid from the initial cubic latticd positional
configuration to steady-state. Sanity check to ensure that we're indeed relaxes for our
given choice of thermalization time T_therm
=#

# ==============================================================================
function mobility_main(dt, Ntherm, M::Int)
    Npart = 10
    T = 1.25
    ρ = 0.6
    γ = 1.0
    rc = 2.5

    L = Npart / cbrt(ρ)
    N = Npart^3
    boundary = CubicBoundary(L, L, L)
    atoms = [Atom(index=i, ϵ=1.0, σ=1.0, mass=1.0) for i in 1:N]

    lj_w_cutoff = LennardJones(cutoff=ShiftedForceCutoff(rc), force_units=NoUnits, energy_units=NoUnits, use_neighbors=true)
    nf = CellListMapNeighborFinder(eligible=trues(N, N), unit_cell=boundary, n_steps=10, dist_cutoff=1.2 * rc)
    simulator = LangevinSplitting(dt=dt, temperature=T, friction=γ, splitting="BAOAB")

    log_freq = 1

    # Arrays to store average observables across replicas
    avg_temp = zeros(Float64, Ntherm+1)
    avg_ke = zeros(Float64, Ntherm+1)
    avg_pe = zeros(Float64, Ntherm+1)
    avg_te = zeros(Float64, Ntherm+1)
    avg_cd = zeros(Float64, Ntherm+1)

    # Variables for storing coordinates and velocities from the first replica
    coords_first_replica = nothing
    vels_first_replica = nothing

    cd_forcing = [(-1.0)^i for i in 1:N] ./ sqrt(N)

    # Run M replicas
    for replica in 1:M
        println("Replica $replica of $M")
        q0 = place_atoms_on_3D_lattice(Npart, L)
        p0 = [random_velocity(1.0, T, 1.0) for _ in 1:N]

        # Loggers
        coord_logger = CoordinateLogger(Float64, log_freq)
        vel_logger = VelocityLogger(Float64, log_freq)
        temp_logger = TemperatureLogger(Float64, log_freq)
        ke_logger = KineticEnergyLogger(Float64, log_freq)
        pe_logger = PotentialEnergyLogger(Float64, log_freq)
        te_logger = TotalEnergyLogger(Float64, log_freq)
        cd_logger = GeneralObservableLogger(colordrift_response, Float64, log_freq)

        loggers = (
            coords = coord_logger,
            vels = vel_logger,
            temp = temp_logger,
            ke = ke_logger,
            pe = pe_logger,
            te = te_logger,
            cd = cd_logger
        )

        sys0 = System(
            atoms = atoms,
            coords = q0,
            velocities = p0,
            boundary = boundary,
            pairwise_inters = (lj_w_cutoff,),
            neighbor_finder = nf,
            loggers = loggers,
            force_units = NoUnits,
            energy_units = NoUnits,
            k = 1.0,
            data = (forcing = cd_forcing,)
        )

        simulate!(sys0, simulator, Ntherm)

        # For the first replica, save the coordinates and velocities
        if replica == 1
            coords_first_replica = sys0.loggers.coords
            vels_first_replica = sys0.loggers.vels
        end

        # Accumulate values for averaging
        avg_temp .+= sys0.loggers.temp.history
        avg_ke .+= sys0.loggers.ke.history
        avg_pe .+= sys0.loggers.pe.history
        avg_te .+= sys0.loggers.te.history
        avg_cd .+= sys0.loggers.cd.history
    end

    # Compute averages
    avg_temp ./= M
    avg_ke ./= M
    avg_pe ./= M
    avg_te ./= M
    avg_cd ./= M

    # Return results
    return (
        coords = coords_first_replica,
        vels = vels_first_replica,
        temp = avg_temp,
        ke = avg_ke,
        pe = avg_pe,
        te = avg_te,
        cd = avg_cd
    )
end

# ==============================================================================
function place_atoms_on_3D_lattice(N::Integer, L)
    reshape([SVector(i*L/N, j*L/N, k*L/N) for i in 0:N-1, j = 0:N-1, k = 0:N-1], N^3)
end

# ==============================================================================
function colordrift_response(s::System, args...; kwargs...)
    p_x = view(reinterpret(reshape, Float64, s.velocities), 1, :)

    return dot(p_x, s.data.forcing)
end

# ==============================================================================
function get_coords(ps)
    x = [p[1] for p in ps]
    y = [p[2] for p in ps]
    z = [p[3] for p in ps]

    return x, y, z
end

## % ============================================================================
t_sim = 2.0
dt_sim = 1e-3
M = 100
N_sim = floor(Int64, t_sim/dt_sim)
tv = [0:N_sim;]*dt_sim
sim_data = mobility_main(dt_sim, N_sim, M)

# %%
coords = values(sim_data.coords)
vels = values(sim_data.vels)
temp = sim_data.temp
ke = sim_data.ke
pe = sim_data.pe
te = sim_data.te
cd = sim_data.cd

# %%
pltEnergy = plot(xaxis="time", yaxis="Energy", title="Energy over time (M = $M)", legend=:outertopright)
plot!(pltEnergy, tv, pe, label="potential")
plot!(pltEnergy, tv, ke, label="kinetic")
plot!(pltEnergy, tv, te, label="total energy")
# display(pltEnergy)

pltTemp = plot(xaxis="time", yaxis="Temperature", title="Temperature over time (M = $M)", legend=:outertopright)
plot!(pltTemp, tv, tv -> 1.25, ls=:dash, color=:black, lw=2, label="Target temp.")
plot!(pltTemp, tv, temp, label="System temp.")
# display(pltTemp)

N_avg = floor(Int64, N_sim/2)
Ete = mean(te[N_avg:end])
rel_err = abs.((te .- Ete)./Ete)*100
pltRE = plot(xaxis="time", yaxis="Relative error", title="Total energy relative error (M = $M)", legend=:outertopright)
plot!(pltRE, tv, rel_err, label="")
# display(pltRE)

pltF = plot(pltEnergy, pltTemp, pltRE, layout=(3,1), size=(800, 600))
display(pltF)

# savefig(pltF, "averages_over_time.pdf")

# %%
coordx, coordy, coordz = get_coords(coords[end])
pltCoords = scatter3d()
scatter3d!(pltCoords, coordx, coordy, coordz, markersize=1, label="Final configuration")
display(pltCoords)

# %%
a = Animation()
for i in 1:N_sim+1
    print("\rstep $i of $(N_sim+1)")
    pltCoordsGif = scatter3d(xlims=(0, 12), ylims=(0, 12), zlims=(0, 12), legend=nothing, title = @sprintf("Time = %.3f", round(i*dt_sim, digits=3)))
    coordx, coordy, coordz = get_coords(coords[i])
    scatter3d!(pltCoordsGif, coordx, coordy, coordz, markersize=1)

    mod(i, 10) == 0 && frame(a, pltCoordsGif)
end

# gif(a, "LJ_relaxation.gif")#, fps=fps)