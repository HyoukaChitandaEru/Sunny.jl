using Sunny, LinearAlgebra, GLMakie, Optim

# Establish geometry of the unit cell.
# "P1" is required due to the rotational symmetry about the
# x-axis being broken.
chain_spacing = 10. # Angstrom
latvecs = chain_spacing * I(3)
one_dimensional_chain = Crystal(latvecs,[[0,0,0]],"P1")

# Establish geometry of the whole chain.
chain_length = 16 # Number of atoms
latsize = (chain_length,1,1) # 1D chain is Nx1x1 lattice
spin_one_chain = System(one_dimensional_chain, latsize, [SpinInfo(1,S=1,g=2)], :SUN)

# Scalar J indicates J*(Sᵢ⋅Sⱼ)
J_groundtruth = -1.

# Interaction is with the left and right next neighbor along the chain (x-direction)
nearest_neighbor_right = Bond(1,1,(1,0,0))
nearest_neighbor_left = Bond(1,1,(-1,0,0))

set_exchange!(spin_one_chain,J_groundtruth,nearest_neighbor_right)
set_exchange!(spin_one_chain,J_groundtruth,nearest_neighbor_left)

D_groundtruth = 10.
Sz = spin_operators(spin_one_chain, 1)[3]
set_onsite_coupling!(spin_one_chain, -D_groundtruth*Sz^2, 1)

Δt = 0.05/D_groundtruth
λ = 0.1
kT = 0.1
langevin = Langevin(Δt; kT, λ);

function viz_chain(sys;kwargs...)#hide
  ##ups = map(x -> abs2(x[1]), sys.coherents)[:];#hide
  ##zs = map(x -> abs2(x[2]), sys.coherents)[:];#hide
  ##downs = map(x -> abs2(x[3]), sys.coherents)[:];#hide
###hide
  ##f = Figure()#hide
  ##ax = LScene(f[1,1];show_axis = false)#hide
  ##_ = Makie.cam3d!(ax.scene, projectiontype=Makie.Orthographic)#hide
###hide
  ##linewidth = 5.#hide
  ##arrowsize = 10.#hide
  ##lengthscale = 15.#hide
  ##pts = [Point3f(Sunny.global_position(sys,site)) for site in eachsite(sys)][:]#hide
###hide
  #### Ups#hide
  ##vecs = [Vec3f([0,0,1]) for site in eachsite(sys)][:]#hide
  ##cols = map(x -> (:blue,x), ups)#hide
  ##Makie.arrows!(ax, pts .+ 0.5 .* vecs, vecs;#hide
        ##linecolor = cols, arrowcolor = cols,#hide
        ##lengthscale, arrowsize, linewidth, kwargs...)#hide
###hide
  #### Downs#hide
  ##vecs = [Vec3f([0,0,-1]) for site in eachsite(sys)][:]#hide
  ##cols = map(x -> (:red,x), downs)#hide
  ##Makie.arrows!(ax, pts .+ 0.5 .* vecs, vecs;#hide
        ##linecolor = cols, arrowcolor = cols,#hide
        ##lengthscale, arrowsize, linewidth, kwargs...)#hide
###hide
  ##cols = map(x -> (:green,x), zs)#hide
  ##meshscatter!(ax,pts, markersize = 7., color = cols)#hide
  ##f#hide
  Sunny.Plotting.plot_coherents(sys;quantization_axis = [0,0,1],kwargs...)
end#hide
randomize_spins!(spin_one_chain)
viz_chain(spin_one_chain)

nStep = 50_000#hide
for _ in 1:nStep#hide
    step!(spin_one_chain, langevin)#hide
end#hide
# ... thermalize ...
viz_chain(spin_one_chain)

sc = dynamical_correlations(spin_one_chain; Δt, nω = 80, ωmax = 20.);#hide

for _ in 1:10_000#hide
    step!(spin_one_chain, langevin)#hide
end#hide
add_sample!(sc, spin_one_chain)#hide
# ... some time later ...
viz_chain(spin_one_chain)

for _ in 1:10_000#hide
    step!(spin_one_chain, langevin)#hide
end#hide
add_sample!(sc, spin_one_chain)#hide
# ... some time later ...
viz_chain(spin_one_chain)

for _ in 1:20#hide
    for _ in 1:10_000#hide
        step!(spin_one_chain, langevin)#hide
    end#hide
    add_sample!(sc, spin_one_chain)#hide
end#hide
# ... some time later ...
viz_chain(spin_one_chain)

sc

SIMULATED_EXPERIMENT_HISTOGRAM_PARAMS = unit_resolution_binning_parameters(sc)

formula = intensity_formula(sc,:perp;kT)
is, counts = intensities_binned(sc,SIMULATED_EXPERIMENT_HISTOGRAM_PARAMS,formula)

SIMULATED_EXPERIMENT_DATA = (is ./ counts)[:,1,1,:]

bcs = axes_bincenters(SIMULATED_EXPERIMENT_HISTOGRAM_PARAMS)
f = Figure()#hide
ax = Axis(f[1,1])#hide
heatmap!(ax,bcs[1],bcs[4],log10.(SIMULATED_EXPERIMENT_DATA))
f#hide

# Same as before
chain_spacing = 10. # Angstrom
latvecs = chain_spacing * I(3)
one_dimensional_chain = Crystal(latvecs,[[0,0,0]],"P1")
chain_length = 16 # Number of atoms
latsize = (chain_length,1,1) # 1D chain is Nx1x1 lattice
spin_one_chain = System(one_dimensional_chain, latsize, [SpinInfo(1,S=1,g=2)], :SUN)

Δt = 0.05
λ = 0.1
kT = 0. # LSWT uses zero temperature
langevin = Langevin(Δt; kT, λ);

nearest_neighbor_right = Bond(1,1,(1,0,0))
nearest_neighbor_left = Bond(1,1,(-1,0,0))

Sz = spin_operators(spin_one_chain, 1)[3]

function forward_problem(J_trial, D_trial)

  # Ensure there is no phase transition (or else LSWT will throw errors)
  J_trial = min(J_trial,0)
  D_trial = max(D_trial,0)

  # Uses J_trial
  set_exchange!(spin_one_chain,J_trial,nearest_neighbor_right)
  set_exchange!(spin_one_chain,J_trial,nearest_neighbor_left)

  # Uses D_trial
  set_onsite_coupling!(spin_one_chain, -D_trial*Sz^2, 1)

  # Perform spin wave calculation, continued below...

  # ... perform spin wave calculation, continued from above.
  one_site_system = reshape_supercell(spin_one_chain,[1 0 0; 0 1 0; 0 0 1])

  langevin.kT = 0.
  nStep = 1_000
  for _ in 1:nStep
      step!(one_site_system, langevin)
  end

  swt = SpinWaveTheory(one_site_system)
  formula = intensity_formula(swt,:perp; kernel = lorentzian(0.5))
  params = SIMULATED_EXPERIMENT_HISTOGRAM_PARAMS
  is_swt = Sunny.intensities_bin_centers(swt, params, formula)

  return is_swt[:,1,1,:]
end # end of forward_problem

function plot_forward(J,D)
  is_swt = forward_problem(J,D)
  bcs = axes_bincenters(SIMULATED_EXPERIMENT_HISTOGRAM_PARAMS)
  heatmap(bcs[1],bcs[4],log10.(is_swt))
end

plot_forward(-1,10)

plot_forward(-6,2)

plot_forward(-0.01,15)

function get_loss(parameters)
  J,D = parameters
  is_swt = forward_problem(J,D)
  sqrt(sum(abs2.(SIMULATED_EXPERIMENT_DATA .- is_swt)))
end

nJ = 30
nD = 35
loss_landscape = zeros(Float64,nJ,nD)
Js = range(-2,0,length=nJ)
Ds = range(8,12,length=nD)
for (ij,J) in enumerate(Js)
  for (id,D) in enumerate(Ds)
    loss_landscape[ij,id] = get_loss([J,D])
  end
end

fig = Figure()
ax = Axis(fig[1,1],xlabel = "J [meV]", ylabel = "D [meV]")
contourf!(ax,Js,Ds,loss_landscape)

x0 = [-2,9.5]
opt_result = optimize(get_loss,x0,method=GradientDescent(alphaguess=1e-3),store_trace=true,extended_trace = true,time_limit=10.)
lines!(ax,Point2f.(Optim.x_trace(opt_result)))
scatter!(ax,-1,10)
fig

bcs = axes_bincenters(SIMULATED_EXPERIMENT_HISTOGRAM_PARAMS)
f = Figure()#hide
ax = Axis(f[1,1]; xlabel="Q [R.L.U.]", ylabel="Energy (meV)")#hide
heatmap!(ax,bcs[1],bcs[4],log10.(SIMULATED_EXPERIMENT_DATA), colormap = :deepsea)
f#hide


J_trial, D_trial = opt_result.minimizer
set_exchange!(spin_one_chain,J_trial,nearest_neighbor_right)#hide
set_exchange!(spin_one_chain,J_trial,nearest_neighbor_left)#hide

set_onsite_coupling!(spin_one_chain, -D_trial*Sz^2, 1)#hide
one_site_system = reshape_supercell(spin_one_chain,[1 0 0; 0 1 0; 0 0 1])#hide

langevin.kT = 0.#hide
nStep = 1_000#hide
for _ in 1:nStep#hide
    step!(one_site_system, langevin)#hide
end#hide

swt = SpinWaveTheory(one_site_system)#hide
params = SIMULATED_EXPERIMENT_HISTOGRAM_PARAMS

path = [[q,0,0] for q in bcs[1]]
disp, intensity = intensities_bands(swt, path, intensity_formula(swt,:perp, kernel = delta_function_kernel))

for i in axes(disp)[2]
    lines!(ax, bcs[1], disp[:,i]; color=intensity[:,i], colormap = :turbo,linewidth = 5,colorrange = (0.,1.))
end
Colorbar(f[1,2],colormap = :turbo, limits = (0.,1.))
Colorbar(f[1,3],colormap = :deepsea, limits = (0.,1.))
f
