
__precompile__(true)

module CompressedSensing

import JuMP
using Clp
using LossFunctions

export l1reconstruct
export evaluateloss

"""
    l1reconstruct(x, A)

Return estimated x by ``\ell_1`` reconstruction.
"""
function l1reconstruct(unknown_x::Array{Float64, 1}, A::Array{Float64, 2})
    N = length(unknown_x)
    if size(A, 2) ≠ N
        error("doesn't match the size of x and A")
    end
    M = size(A, 1)
    y = A * unknown_x

    l1 = JuMP.Model(solver=ClpSolver())
    JuMP.@variable(l1, t[1:N])
    JuMP.@variable(l1, x[1:N] >= 0.0)

    JuMP.@objective(l1, Min, sum(t[n] for n=1:N))

    JuMP.@constraint(l1, [n=1:N], -t[n] <= x[n])
    JuMP.@constraint(l1, [n=1:N], x[n] <= t[n])
    JuMP.@constraint(l1, [m=1:M], y[m] == sum(A[m,n] * x[n] for n=1:N))
    status = JuMP.solve(l1)
    JuMP.getvalue(x)
end

"""
    evaluateloss(x, A; alg, loss)

Return evaluated loss by an algorithm with a loss.
Default alg=l1reconstruct, loss=L2DisctLoss().
"""
function evaluateloss(x::Array{Float64, 1}, A::Array{Float64, 2}; alg=l1reconstruct, loss=L2DistLoss())
    estimatedx = alg(x, A)
    loss(estimated_x, A)
end

end # module
