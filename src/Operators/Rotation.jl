# Magnitude of axis n is ignored. Angle θ in radians.
function axis_angle_to_matrix(n, θ)
    x, y, z = normalize(n)
    s, c = sincos(θ)
    t = 1 - c
    return SA[t*x*x+c    t*x*y-z*s  t*x*z+y*s
              t*x*y+z*s  t*y*y+c    t*y*z-x*s
              t*x*z-y*s  t*y*z+x*s  t*z*z+c]
end

function matrix_to_axis_angle(R::Mat3)
    # Assertion disabled for performance
    # @assert R'*R ≈ I && det(R) ≈ 1

    # Formula derived by Mike Day, Insomniac Games, and posted online as
    # "Converting a Rotation Matrix to a Quaternion".
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    (m00, m10, m20, m01, m11, m21, m02, m12, m22) = R[:]
    if (m22 < 0)
        if (m00 > m11)
            t = 1 + m00 - m11 - m22
            q = SA[t, m01+m10, m20+m02, m12-m21]
        else
            t = 1 - m00 + m11 - m22
            q = SA[m01+m10, t, m12+m21, m20-m02]
        end
    else
        if (m00 < -m11)
            t = 1 - m00 - m11 + m22
            q = SA[m20+m02, m12+m21, t, m01-m10]
        else
            t = 1 + m00 + m11 + m22
            q = SA[m12-m21, m20-m02, m01-m10, t]
        end
    end

    # Construct a unit quaternion
    q *= 0.5 / sqrt(t)

    # Angle of rotation
    q4 = max(min(q[4], 1.0), -1.0)
    θ = 2acos(q4)

    if θ < 1e-12
        # Axis is ill-defined for the identity matrix, but we don't want NaNs
        n = SA[0., 0., 0.]
    else
        # Standard conversion from a unit quaternion q to an axis-angle
        n = SA[q[1], q[2], q[3]] / sqrt(1 - q[4]^2)
    end

    # Negate the axis to invert the rotation, i.e., transpose R. This is
    # necessary to view R as right-multiplying a column vector.
    n = -n

    return (n, θ)
end

# Generate a random, orthogonal NxN matrix under the Haar measure
function random_orthogonal(rng, N::Int; special=false)
    # This approach is simple and correct as described below:
    # https://math.stackexchange.com/q/2166210/660903
    # More efficient methods are discussed here:
    # https://doi.org/10.1137/0908055
    # https://arxiv.org/abs/math-ph/0609050
    (; U, V) = svd(randn(rng, Float64, N,N))
    O = U*V'
    return special ? O*det(O) : O
end

# Unitary for a rotation matrix built from abstract generators.
function unitary_for_rotation(R::Mat3, gen)
    !(R'*R ≈ I)   && error("Not an orthogonal matrix, R = $R.")
    !(det(R) ≈ 1) && error("Matrix includes a reflection, R = $R.")
    n, θ = matrix_to_axis_angle(R)
    return exp(-im*θ*(n'*gen))
end

# Unitary for a rotation matrix in the N-dimensional irrep of SU(2).
function unitary_irrep_for_rotation(R::Mat3; N::Int)
    gen = spin_matrices_of_dim(; N)
    unitary_for_rotation(R, gen)
end

# Unitary for a rotation matrix in the (N1⊗N2⊗...)-dimensional irrep of SU(2).
function unitary_tensor_for_rotation(R::Mat3; Ns)
    Us = [unitary_irrep_for_rotation(R; N) for N in Ns]
    if length(Ns) == 1
        return Us[1]
    elseif length(Ns) == 2
        return kron(Us[1], Us[2])
    else
        error("Tensor products currently accept only two operators")
    end
end

# TODO: Replace this with a function that takes generators.
"""
    rotate_operator(A, R)

Rotates the local quantum operator `A` according to the ``3×3`` rotation matrix
`R`.
"""
function rotate_operator(A::HermitianC64, R)
    isempty(A) && return A
    R = convert(Mat3, R)
    N = size(A, 1)
    U = unitary_irrep_for_rotation(R; N)
    return Hermitian(U'*A*U)
end


# Given an operator A in a tensor product representation labeled by (N1, N2,
# ...), perform a physical rotation by the matrix R .
function rotate_tensor_operator(A::Matrix, R; Ns)
    U = unitary_tensor_for_rotation(R; Ns)
    return U'*A*U
end
