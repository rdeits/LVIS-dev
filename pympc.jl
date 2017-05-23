module PyMPC

using PyCall
push!(PyVector(pyimport("sys")["path"]), joinpath(dirname(@__FILE__), "py-mpc"))

@pyimport pympc.geometry as geometry
@pyimport pympc.control as control
@pyimport pympc.dynamical_systems as dynamical_systems
@pyimport pympc.plot as mpcplot
@pyimport scipy.spatial as spatial
@pyimport pympc.optimization.mpqpsolver as mpqpsolver

using JuMP
using Gurobi

function solve_qp(qp::PyObject, x::AbstractVector)
    if !haskey(qp, :G)
        qp[:remove_linear_terms]()
    end

    G = qp[:G]
    W = qp[:W]
    S = qp[:S]
    H = qp[:H]
    H_inv = qp[:H_inv]
    
    m = Model(solver=GurobiSolver(OutputFlag=0))
    @variable m z[1:size(G, 2)]
    @objective m Min (z' * H * z)[1]
    @constraint m constrs G * z .<= S * x + W
    status = solve(m, suppress_warnings=true)
    if status == :Optimal
        lambdas = vec(getdual.(constrs))
        active_set = abs.(lambdas) .>= 1e-4

        G_A = G[active_set, :]
        W_A = W[active_set, :]
        S_A = S[active_set, :]
        G_I = G[!active_set, :]
        W_I = W[!active_set, :]
        S_I = S[!active_set, :]
        H_A = inv(G_A * H_inv * G_A')

        lambda_A_offset = -H_A * W_A
        lambda_A_linear = -H_A * S_A

        z_offset = -H_inv * G_A' * lambda_A_offset
        z_linear = -H_inv * G_A' * lambda_A_linear

        u_offset = z_offset - H_inv * qp[:F_u]
        u_linear = z_linear - H_inv * qp[:F_xu]'

        status, u_offset .+ u_linear * x, u_linear
    else
        status, fill(NaN, size(G, 2), 1), fill(NaN, size(G, 2), length(x))
    end
end

end
