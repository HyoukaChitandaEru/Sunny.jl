using Sunny, GLMakie

lat_vecs = Sunny.lattice_vectors(1.0, 1.0, 2.0, 90, 90, 120)
basis_vecs = [[0,0,0]]
cryst = Crystal(lat_vecs, basis_vecs)

L = 40
dims = (L, L, 1)
sys = System(cryst, dims, [SpinInfo(1, S=1, g=1)], :SUN; seed=101, units=Units.theory)

J1 = -1           # Nearest-neighbor ferromagnetic
J2 = (2.0/(1+√5)) # Tune competing exchange to set skyrmion scale length
Δ = 2.6           # Exchange anisotropy

ex1 = J1 * [1.0 0.0 0.0;
            0.0 1.0 0.0;
            0.0 0.0 Δ]
ex2 = J2 * [1.0 0.0 0.0;
            0.0 1.0 0.0;
            0.0 0.0 Δ]
set_exchange!(sys, ex1, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys, ex2, Bond(1, 1, [1, 2, 0]));

h = 15.5
field = set_external_field!(sys, [0.0 0.0 h]);

D = 19.0
Sz = Sunny.spin_operators(sys, 1)[3]
set_onsite_coupling!(sys, D*Sz^2, 1);
sys#hide

randomize_spins!(sys)

Δt = 0.2/D  # Integration time step (inverse meV). Typically this will be
            # inversely proportional to the largest energy scale in the
            # system. We can use a fairly large time-step here because
            # accuracy isn't critical.
kT = 0      # Target equilibrium temperature (meV)
λ = 0.1     # Magnitude of coupling to thermal bath (dimensionless)
integrator = Langevin(Δt; kT, λ);

τs = [4., 16, 256]  # Times to record snapshots
frames = []         # Empty array to store snapshots
for i in eachindex(τs)
    dur = i == 1 ? τs[1] : τs[i] - τs[i-1] # Determine the length of time to simulate
    numsteps = round(Int, dur/Δt)
    for _ in 1:numsteps                    # Perform the integration
        step!(sys, integrator)
    end
    push!(frames, copy(sys.coherents))     # Save a snapshot spin configuration
end

include(joinpath(pkgdir(Sunny), "examples", "extra", "plotting2d.jl"))

function sun_berry_curvature(z₁, z₂, z₃)
    z₁, z₂, z₃ = normalize.((z₁, z₂, z₃))
    n₁ = z₁ ⋅ z₂
    n₂ = z₂ ⋅ z₃
    n₃ = z₃ ⋅ z₁
    return angle(n₁ * n₂ * n₃)
end

plot_triangular_plaquettes(sun_berry_curvature, frames; resolution=(1800,600),
    offset_spacing=10, texts = ["\tt = "*string(τ) for τ in τs], text_offset = (0.0, 6.0)
)
