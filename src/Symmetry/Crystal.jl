struct SiteSymmetry
    symbol       :: String
    multiplicity :: Int
    wyckoff      :: Char
end


"""
An object describing a crystallographic unit cell and its space group symmetry.
Constructors are as follows:


    Crystal(filename; symprec=1e-5)

Reads the crystal from a `.cif` file located at the path `filename`.  The
optional parameter `symprec` controls the precision tolerance for spacegroup
symmetries.

    Crystal(latvecs, positions; types=nothing, symprec=1e-5)

Constructs a crystal from the complete list of atom positions `positions`, with
coordinates (between 0 and 1) in units of lattice vectors `latvecs`. Spacegroup
symmetry information is automatically inferred. The optional parameter `types`
is a list of strings, one for each atom, and can be used to break
symmetry-equivalence between atoms.

    Crystal(latvecs, positions, spacegroup_number; types=nothing, setting=nothing, symprec=1e-5)

Builds a crystal by applying symmetry operators for a given international
spacegroup number. For certain spacegroups, there are multiple possible unit
cell settings; in this case, a warning message will be printed, and a list of
crystals will be returned, one for every possible setting. Alternatively, the
optional `setting` string will disambiguate between unit cell conventions.

Currently, crystals built using only the spacegroup number will be missing some
symmetry information. It is generally preferred to build a crystal from a `.cif`
file or from the full specification of the unit cell.


# Examples

```julia
# Read a Crystal from a .cif file
Crystal("filename.cif")

# Build an FCC crystal using the primitive unit cell. The spacegroup number
# 225 is inferred.
latvecs = [1 1 0;
            1 0 1;
            0 1 1] / 2
positions = [[0, 0, 0]]
Crystal(latvecs, positions)

# Build a CsCl crystal (two cubic sublattices). By providing distinct type
# strings, the spacegroup number 221 is inferred.
latvecs = lattice_vectors(1, 1, 1, 90, 90, 90)
positions = [[0,0,0], [0.5,0.5,0.5]]
types = ["Na", "Cl"]
cryst = Crystal(latvecs, positions; types)

# Build a diamond cubic crystal from its spacegroup number 227. This
# spacegroup has two possible settings ("1" or "2"), which determine an
# overall unit cell translation.
latvecs = lattice_vectors(1, 1, 1, 90, 90, 90)
positions = [[1, 1, 1] / 4]
cryst = Crystal(latvecs, positions, 227; setting="1")
```

See also [`lattice_vectors`](@ref).
"""
struct Crystal
    root           :: Union{Nothing, Crystal}              # Root crystal (invariant under `subcrystal` and reshaping)
    latvecs        :: Mat3                                 # Lattice vectors as columns
    prim_latvecs   :: Mat3                                 # Primitive lattice vectors
    recipvecs      :: Mat3                                 # Reciprocal lattice vectors (conventional)
    positions      :: Vector{Vec3}                         # Positions in fractional coords
    types          :: Vector{String}                       # Types
    classes        :: Vector{Int}                          # Class indices
    sitesyms       :: Union{Nothing, Vector{SiteSymmetry}} # Optional site symmetries
    symops         :: Vector{SymOp}                        # Symmetry operations
    spacegroup     :: String                               # Description of space group
    symprec        :: Float64                              # Tolerance to imperfections in symmetry
end

# Constructs a crystal from the complete list of atom positions `positions`,
# representing fractions (between 0 and 1) of the lattice vectors `latvecs`.
# All symmetry information is automatically inferred.
function Crystal(latvecs, positions; types::Union{Nothing,Vector{String}}=nothing, symprec=1e-5)
    print_crystal_warnings(latvecs, positions)
    latvecs = convert(Mat3, latvecs)
    positions = [convert(Vec3, p) for p in positions]
    if isnothing(types)
        types = fill("", length(positions))
    end
    return crystal_from_inferred_symmetry(latvecs, positions, types; symprec)
end


# Builds a crystal by applying the symmetry operators for a given spacegroup
# symbol.
function Crystal(latvecs, positions, symbol::String; types::Union{Nothing,Vector{String}}=nothing, setting=nothing, symprec=1e-5)
    print_crystal_warnings(latvecs, positions)
    latvecs = convert(Mat3, latvecs)
    positions = [convert(Vec3, p) for p in positions]
    if isnothing(types)
        types = fill("", length(positions))
    end
    crystal_from_symbol(latvecs, positions, types, symbol; setting, symprec)
end

# Builds a crystal by applying symmetry operators for a given international
# spacegroup number.
function Crystal(latvecs, positions, spacegroup_number::Int; types::Union{Nothing,Vector{String}}=nothing, setting=nothing, symprec=1e-5)
    print_crystal_warnings(latvecs, positions)
    latvecs = convert(Mat3, latvecs)
    positions = [convert(Vec3, p) for p in positions]
    if isnothing(types)
        types = fill("", length(positions))
    end
    symbol = string(spacegroup_number)
    crystal_from_symbol(latvecs, positions, types, symbol; setting, symprec)
end

function print_crystal_warnings(latvecs, positions)
    det(latvecs) < 0 && @warn "Lattice vectors are not right-handed."
    if length(positions) >= 100
        @info """This a very large crystallographic cell, which Sunny does not handle well.
                 If the intention is to model chemical inhomogeneity, the recommended procedure is as
                 follows: First, create a small unit cell with an idealized structure. Next, create
                 a perfectly periodic `System` of the desired size. Finally, use `to_inhomogeneous`
                 and related functions to design a system with the desired inhomogeneities."""
    end
end

"""
    natoms(cryst::Crystal)

Number of atoms in the unit cell, i.e., number of Bravais sublattices.
"""
@inline natoms(cryst::Crystal) = length(cryst.positions)

"""
    cell_volume(cryst::Crystal)

Volume of the crystal unit cell.
"""
cell_volume(cryst::Crystal) = abs(det(cryst.latvecs))


function spacegroup_name(hall_number::Int)
    # String representation of space group
    sgt = Spglib.get_spacegroup_type(hall_number)
    return "'$(sgt.international)' ($(sgt.number))"
end


# Sort the sites according to class and fractional coordinates.
function sort_sites!(cryst::Crystal)
    function less_than(i, j)
        ci = cryst.classes[i]
        cj = cryst.classes[j]
        if ci != cj
            return ci < cj
        end
        ri = cryst.positions[i]
        rj = cryst.positions[j]
        for k = 3:-1:1
            if !isapprox(ri[k], rj[k], atol=10cryst.symprec)
                return ri[k] < rj[k]
            end
        end
        str1, str2 = fractional_vec3_to_string.((ri, rj))
        error("""Detected two very close atoms ($str1 and $str2).
                 If positions inferred from spacegroup, try increasing `symprec` parameter to `Crystal`.""")
    end
    perm = sort(eachindex(cryst.positions), lt=less_than)
    cryst.positions .= cryst.positions[perm]
    cryst.classes .= cryst.classes[perm]
    cryst.types .= cryst.types[perm]
end


function crystal_from_inferred_symmetry(latvecs::Mat3, positions::Vector{Vec3}, types::Vector{String}; symprec=1e-5)
    # Print a warning if non-conventional lattice vectors are detected.
    try cell_type(latvecs) catch e @warn e.msg end

    for i in 1:length(positions)
        for j in i+1:length(positions)
            ri = positions[i]
            rj = positions[j]
            if all_integer(ri-rj; symprec)
                error("Positions $ri and $rj are symmetry equivalent.")
            end
        end
    end

    recipvecs = 2π*Mat3(inv(latvecs)')
    positions = wrap_to_unit_cell.(positions; symprec)

    cell = Spglib.Cell(latvecs, positions, types)
    d = Spglib.get_dataset(cell, symprec)
    classes = d.crystallographic_orbits
    # classes = d.equivalent_atoms
    symops = SymOp.(d.rotations, d.translations)
    spacegroup = spacegroup_name(Int(d.hall_number))

    # renumber class indices so that they go from 1:max_class
    classes = [findfirst(==(c), unique(classes)) for c in classes]
    @assert unique(classes) == 1:maximum(classes)

    # multiplicities for the equivalence classes
    multiplicities = map(classes) do c
        # atoms that belong to class c
        atoms = findall(==(c), classes)
        # atoms in the primitive cell that belong to class c
        prim_atoms = unique(d.mapping_to_primitive[atoms])
        # number of atoms in the standard cell that correspond to each primitive index for class c
        counts = [count(==(i), d.std_mapping_to_primitive) for i in prim_atoms]
        # sum over all equivalent atoms in the primitive cell
        sum(counts)
    end

    sitesyms = SiteSymmetry.(d.site_symmetry_symbols, multiplicities, d.wyckoffs)

    ret = Crystal(nothing, latvecs, d.primitive_lattice, recipvecs, positions, types, classes, sitesyms, symops, spacegroup, symprec)
    validate(ret)
    return ret
end


# Build Crystal using the space group denoted by a unique Hall number. The complete
# list is given at http://pmsl.planet.sci.kobe-u.ac.jp/~seto/?page_id=37&lang=en
function crystal_from_hall_number(latvecs::Mat3, positions::Vector{Vec3}, types::Vector{String}, hall_number::Int; symprec=1e-5)
    cell = cell_type(latvecs)
    hall_cell = cell_type(hall_number)
    allowed_cells = all_compatible_cells(hall_cell)
    @assert cell in allowed_cells "Hall number $hall_number requires a $hall_cell cell, but found $cell."

    if hall_cell == Sunny.monoclinic
        is_compatible = is_compatible_monoclinic_cell(latvecs, hall_number)
        @assert is_compatible "Lattice vectors define a monoclinic cell that is incompatible with Hall number $hall_number."
    end

    rotations, translations = Spglib.get_symmetry_from_database(hall_number)
    symops = SymOp.(rotations, translations)
    spacegroup = spacegroup_name(hall_number)

    return crystal_from_symops(latvecs, positions, types, symops, spacegroup; symprec)
end

function crystal_from_symbol(latvecs::Mat3, positions::Vector{Vec3}, types::Vector{String}, symbol::String; setting=nothing, symprec=1e-5)
    hall_numbers = Int[]
    crysts = Crystal[]

    n_hall_numbers = 530
    for hall_number in 1:n_hall_numbers
        sgt = Spglib.get_spacegroup_type(hall_number)

        if (replace(symbol, " "=>"") == sgt.international_short || 
            symbol in [string(sgt.number), sgt.hall_symbol, sgt.international, sgt.international_full])

            # Some Hall numbers may be incompatible with unit cell of provided
            # lattice vectors; skip them.
            is_compatible = true

            cell = cell_type(latvecs)
            hall_cell = cell_type(hall_number)
            allowed_cells = all_compatible_cells(hall_cell)

            # Special handling of trigonal space groups
            if Sunny.is_trigonal_symmetry(hall_number)
                # Trigonal symmetry must have either hexagonal or rhombohedral
                # cell, according to the Hall number.
                is_latvecs_valid = cell in [Sunny.rhombohedral, Sunny.hexagonal]
                if !is_latvecs_valid
                    error("Symbol $symbol requires a rhombohedral or hexagonal cell, but found $cell.")
                end
                is_compatible = cell in allowed_cells
            else
                # For all other symmetry types, there is a unique cell for each Hall number
                if !(cell in allowed_cells)
                    error("Symbol $symbol requires a $hall_cell cell, but found $cell.")
                end
            end

            if hall_cell == Sunny.monoclinic
                is_compatible = is_compatible_monoclinic_cell(latvecs, hall_number)
            end

            if is_compatible
                cryst = crystal_from_hall_number(latvecs, positions, types, hall_number; symprec)
                push!(hall_numbers, hall_number)
                push!(crysts, cryst)
            end
        end
    end

    if length(crysts) == 0
        error("Could not find symbol '$symbol' in database.")
    elseif length(crysts) == 1
        return first(crysts)
    else
        if !isnothing(setting)
            i = findfirst(hall_numbers) do hall_number
                sgt = Spglib.get_spacegroup_type(hall_number)
                setting == sgt.choice
            end
            if isnothing(i)
                error("The symbol '$symbol' is ambiguous, and the specified setting '$setting' is not valid.")
            else
                return crysts[i]
            end
        end

        # TODO: Make this @warn
        println("The spacegroup '$symbol' allows for multiple settings!")
        println("Returning a list of the possible crystals:")
        for (i, (hall_number, c)) in enumerate(zip(hall_numbers, crysts))
            sgt = Spglib.get_spacegroup_type(hall_number)
            hm_symbol = sgt.international
            choice = sgt.choice
            n_atoms = length(c.positions)
            i_str = @sprintf "%2d" i
            natoms_str = @sprintf "%2d" n_atoms
            println("   $i_str. \"$hm_symbol\", setting=\"$choice\", with $natoms_str atoms")
        end
        println()
        println("Note: To disambiguate, you may pass a named parameter, setting=\"...\".")
        println()
        return crysts
    end
end

# Builds a crystal from an explicit set of symmetry operations and a minimal set of positions
function crystal_from_symops(latvecs::Mat3, positions::Vector{Vec3}, types::Vector{String}, symops::Vector{SymOp}, spacegroup::String; symprec=1e-5)
    all_positions = Vec3[]
    all_types = String[]
    classes = Int[]
    
    for i = eachindex(positions)
        for s = symops
            x = wrap_to_unit_cell(transform(s, positions[i]); symprec)

            j = findfirst(y -> all_integer(x-y; symprec), all_positions)
            if isnothing(j)
                push!(all_positions, x)
                push!(all_types, types[i])
                push!(classes, i)
            else
                j_ref = classes[j]
                if i != j_ref
                    error("Reference positions $(positions[i]) and $(positions[j_ref]) are symmetry equivalent.")
                end
            end
        end
    end

    # Atoms are sorted by contiguous equivalence classes: 1, 2, ..., n
    @assert unique(classes) == 1:maximum(classes)

    # Ask Spglib to infer the spacegroup for the given positions and types
    inferred = crystal_from_inferred_symmetry(latvecs, all_positions, all_types; symprec)

    # Compare the inferred symops to the provided ones
    is_subgroup = all(symops) do s
        any(inferred.symops) do s′
            isapprox(s, s′; atol=symprec)
        end
    end
    is_supergroup = all(inferred.symops) do s
        any(symops) do s′
            isapprox(s, s′; atol=symprec)
        end
    end

    if !is_subgroup
        @warn """User provided symmetry operation could not be inferred by Spglib,
                 which likely indicates a non-conventional unit cell."""
    end

    # If the inferred symops match the provided ones, then we use the inferred
    # Crystal. Otherwise we must construct a new Crystal without primitive
    # lattice and site symmetry information.
    ret = if is_subgroup && is_supergroup
        inferred
    else
        prim_latvecs = latvecs
        recipvecs = 2π*Mat3(inv(latvecs)')
        Crystal(nothing, latvecs, prim_latvecs, recipvecs, all_positions, all_types, classes, nothing, symops, spacegroup, symprec)
    end
    sort_sites!(ret)
    validate(ret)
    return ret
end


"""
    primitive_cell_shape(cryst::Crystal)

Returns the shape of the primitive cell as a 3×3 matrix, in fractional
coordinates of the conventional lattice vectors. May be useful for constructing
inputs to [`reshape_supercell`](@ref).

# Examples
```julia
# Valid if `cryst` has not been reshaped
@assert cryst.prim_latvecs ≈ cryst.latvecs * primitive_cell_shape(cryst)
```
"""
function primitive_cell_shape(cryst::Crystal)
    root = @something cryst.root cryst
    if isnothing(root.prim_latvecs)
        error("Primitive lattice vectors not available.")
    end
    return root.latvecs \ root.prim_latvecs
end


function check_shape_commensurate(cryst, shape)
    root = @something cryst.root cryst
    if !isnothing(root.prim_latvecs)
        shape = root.prim_latvecs \ root.latvecs * shape
    end

    # Each of these components must be integer
    if !(round.(shape) ≈ shape)
        if !isnothing(root.prim_latvecs)
            error("Cell shape must be integer multiples of primitive lattice vectors. Calculated $shape.")
        else
            error("Cell shape must be a 3×3 matrix of integers. Received $shape.")
        end
    end
end

function reshape_crystal(cryst::Crystal, new_shape::Mat3)
    # Check that desired lattice vectors are commensurate with root cell
    check_shape_commensurate(cryst, new_shape)

    # The `root.latvecs` defines the fractional coordinate system, but note that
    # `cryst` may be formed as a subcrystal of `root`. For reshaping purposes,
    # therefore, we must use `cryst.positions` and not `root.positions`, etc.
    root = @something cryst.root cryst

    # Lattice vectors of the new unit cell in global coordinates
    new_latvecs = root.latvecs * new_shape

    # Return this crystal if possible
    new_latvecs ≈ cryst.latvecs && return cryst

    # Symmetry precision needs to be rescaled for the new unit cell. Ideally we
    # would have three separate rescalings (one per lattice vector), but we're
    # forced to pick just one. Scale according to volume change.
    new_symprec = root.symprec / cbrt(abs(det(new_shape)))

    # This matrix defines a mapping from fractional coordinates `x` in `cryst`
    # to fractional coordinates `y` in the new unit cell.
    B = new_latvecs \ cryst.latvecs

    # Our goal is to loop over `cryst` cells (n1, n2, n3) and make sure we
    # completely cover `new_latvecs` cell, {0 ≤ y₁ ≤ 1, 0 ≤ y₂ ≤ 1, 0 ≤ y₃ ≤ 1}.
    # Due to the linear relationship `y = Bx`, the required range of (n1, n2,
    # n3) is associated with inv(B). For example, `inv(B) * [1, 1, 1]` should be
    # covered in the space of `x` sampling.
    #
    # It is not clear to me what the right formula is, and the
    # `sum.(eachrow(...))` formula below is a conservative, heuristic guess.
    # Fortunately, it is well protected against mistakes via the check on atom
    # count. Any mistake will yield an assertion error: "Missing atoms in
    # reshaped unit cell".
    nmax = round.(Int, sum.(eachrow(abs.(inv(B))))) .+ 1

    new_positions = Vec3[]
    new_types     = String[]
    new_classes   = Int[]
    new_sitesyms  = isnothing(cryst.sitesyms) ? nothing : SiteSymmetry[]

    for i in 1:natoms(cryst)
        for n1 in -nmax[1]:nmax[1], n2 in -nmax[2]:nmax[2], n3 in -nmax[3]:nmax[3]
            x = cryst.positions[i] + Vec3(n1, n2, n3)
            y = B*x

            # Check whether the new position y (in fractional coordinates
            # associated with `new_latvecs`) is within the new unit cell.
            # Account for finite symprec ϵ by checking the bounds [-ϵ,1-ϵ). See
            # related comment in `wrap_to_unit_cell`.
            if all(-new_symprec .<= y .< 1 - new_symprec)
                push!(new_positions, y)
                push!(new_types, cryst.types[i])
                push!(new_classes, cryst.classes[i])
                !isnothing(cryst.sitesyms) && push!(new_sitesyms, cryst.sitesyms[i])
            end
        end
    end

    # Check that we have the right number of atoms
    @assert abs(det(B)) * length(new_positions) ≈ natoms(cryst) "Missing atoms in reshaped unit cell. Please report this bug!"

    # Reciprocal lattice vectors of the new unit cell
    new_recipvecs = 2π * Mat3(inv(new_latvecs)')

    # Create an empty list of symops as a marker that this information has been
    # lost with the resizing procedure.
    new_symops = SymOp[]

    return Crystal(root, new_latvecs, root.prim_latvecs, new_recipvecs, new_positions, new_types, new_classes,
                   new_sitesyms, new_symops, root.spacegroup, new_symprec)
end


"""
    subcrystal(cryst, types) :: Crystal

Filters sublattices of a `Crystal` by atom `types`, keeping the space group
unchanged.

    subcrystal(cryst, classes) :: Crystal

Filters sublattices of `Crystal` by equivalence `classes`, keeping the space
group unchanged.
"""
function subcrystal(cryst::Crystal, types::Vararg{String, N}) where N
    for s in types
        if !(s in cryst.types)
            error("types string '$s' is not present in crystal.")
        end
    end
    atoms = findall(in(types), cryst.types)
    classes = unique(cryst.classes[atoms])
    return subcrystal(cryst, classes...)
end

function subcrystal(cryst::Crystal, classes::Vararg{Int, N}) where N
    root = @something cryst.root cryst
    for c in classes
        if !(c in cryst.classes)
            error("Class '$c' is not present in crystal.")
        end
    end

    atoms = findall(in(classes), cryst.classes)
    new_positions = cryst.positions[atoms]
    new_types = cryst.types[atoms]
    new_classes = cryst.classes[atoms]
    new_sitesyms = cryst.sitesyms[atoms]

    if atoms != 1:maximum(atoms)
        @info "Atoms have been renumbered in subcrystal."
    end

    ret = Crystal(root, cryst.latvecs, cryst.prim_latvecs, cryst.recipvecs, new_positions, new_types,
                  new_classes, new_sitesyms, cryst.symops, cryst.spacegroup, cryst.symprec)
    return ret
end

# Avoids ambiguity error
subcrystal(cryst::Crystal) = cryst

function Base.show(io::IO, cryst::Crystal)
    spg = isempty(cryst.spacegroup) ? "" : "$(cryst.spacegroup), "
    println(io, "Crystal($(spg)$(natoms(cryst)) atoms)")
end

function Base.show(io::IO, ::MIME"text/plain", cryst::Crystal)
    printstyled(io, "Crystal\n"; bold=true, color=:underline)
    println(io, "Spacegroup $(cryst.spacegroup)")

    if is_standard_form(cryst.latvecs)
        (a, b, c, α, β, γ) = lattice_params(cryst.latvecs)
        @printf io "Lattice params a=%.4g, b=%.4g, c=%.4g, α=%.4g°, β=%.4g°, γ=%.4g°\n" a b c α β γ
    else
        println(io, formatted_matrix(number_to_math_string.(cryst.latvecs); prefix="Lattice vectors: "))
    end

    @printf io "Cell volume %.4g\n" cell_volume(cryst)

    for c in unique(cryst.classes)
        i = findfirst(==(c), cryst.classes)
        descr = String[]
        if cryst.types[i] != ""
            push!(descr, "Type '$(cryst.types[i])'")
        end
        if !isnothing(cryst.sitesyms)
            symbol = cryst.sitesyms[i].symbol
            multiplicity = cryst.sitesyms[i].multiplicity
            wyckoff = cryst.sitesyms[i].wyckoff
            push!(descr, "Wyckoff $multiplicity$wyckoff (point group '$symbol')")
        end
        if isempty(descr)
            push!(descr, "Class $c")
        end
        println(io, join(descr, ", "), ":")

        for i in findall(==(c), cryst.classes)
            pos = fractional_vec3_to_string(cryst.positions[i])
            println(io, "   $i. $pos")
        end
    end
end

function validate(cryst::Crystal)
    # Atoms of the same class must have the same type
    for i in eachindex(cryst.positions)
        for j in eachindex(cryst.positions)
            if cryst.classes[i] == cryst.classes[j]
                @assert cryst.types[i] == cryst.types[j]
            end
        end
    end

    # Rotation matrices in global coordinates must be orthogonal
    for s in cryst.symops
        R = cryst.latvecs * s.R * inv(cryst.latvecs)
        # Due to possible imperfections in the lattice vectors, only require
        # that R is approximately orthogonal
        @assert norm(R*R' - I) < cryst.symprec "Lattice vectors and symmetry operations are incompatible."
    end

    # TODO: Check that space group is closed and that symops have inverse?
end

#= Definitions of common crystals =#

function square_crystal(; a=1.0, c=10a)
    latvecs = lattice_vectors(a, a, c, 90, 90, 90)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions)
    return cryst
end

function triangular_crystal(; a=1.0, c=10a)
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0]]
    cryst = Crystal(latvecs, positions)
    return cryst
end

function kagome_crystal(; a=1.0, c=10a)
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]]
    cryst = Crystal(latvecs, positions)
    sort_sites!(cryst)
    return cryst
end

function hexagonal_crystal(; a=1.0, c=10a)
    latvecs = lattice_vectors(a, a, c, 90, 90, 120)
    positions = [[0, 0, 0], [1/3, 2/3, 0]]
    cryst = Crystal(latvecs, positions)
    sort_sites!(cryst)
    return cryst
end

function cubic_crystal(; a=1.0)
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[0, 0, 0]]
    return Crystal(latvecs, positions)
end

function fcc_crystal(; a=1.0)
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[0, 0, 0]/2,
                 [1, 1, 0]/2,
                 [1, 0, 1]/2,
                 [0, 1, 1]/2]
    cryst = Crystal(latvecs, positions)
    sort_sites!(cryst)
    return cryst
end

function bcc_crystal(; a=1.0)
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[0, 0, 0]/2, [1, 1, 1]/2]
    return Crystal(latvecs, positions)
end

function diamond_crystal(; a=1.0)
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [
        [0, 0, 0]/4,
        [2, 2, 0]/4,
        [1, 1, 1]/4,
        [3, 3, 1]/4,
        [2, 0, 2]/4,
        [0, 2, 2]/4,
        [3, 1, 3]/4,
        [1, 3, 3]/4,
    ]
    cryst = Crystal(latvecs, positions)
    sort_sites!(cryst)
    return cryst
end

function pyrochlore_crystal(; a=1.0)
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    return Crystal(latvecs, [[0, 0, 0]], 227, setting="2")
end

function hyperkagome_crystal(; a=1.0)
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    # This shift is appropriate to the Ir atoms of a conventional Na4Ir3O8 cell,
    # but any other shift would yield the same symmetries
    # https://materials.springer.com/isp/crystallographic/docs/sd_1723194
    x = 0.141
    p = [1/8, x, x+1/4]
    cryst = Crystal(latvecs, [p], 213)
    @assert !isnothing(cryst.sitesyms)
    return cryst
end
