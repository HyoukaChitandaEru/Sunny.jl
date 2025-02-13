using Sunny, GLMakie

# Square lattice
latvecs = lattice_vectors(1, 1, 10, 90, 90, 90)
cryst = Crystal(latvecs, [[0,0,0]])

# Simple Heisenberg model
sys = System(cryst, (10,10,1), [SpinInfo(1, S=1, g=2)], :dipole; seed=1)
J = -1.0
set_exchange!(sys, J, Bond(1, 1, (1, 0, 0)))
randomize_spins!(sys)

fig = plot_spins(sys; colorfn=i->sys.dipoles[i][3], colorrange=(-1, 1), dims=2)

Δt = 0.1/abs(J)
langevin = Langevin(Δt; kT=0, λ=0.05)

# View an animation in real time
for _ in 1:500
    for _ in 1:5
        step!(sys, langevin)
    end
    notify(fig)
    sleep(1/60)
end

# Save an animation to file
randomize_spins!(sys)
record(fig, "animation.mp4", 1:500; framerate=30) do _
    for _ in 1:5
        step!(sys, langevin)
    end
    notify(fig)
end
