module PyMPC

using DrakeVisualizer
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

colmat(x::AbstractArray) = reshape(x, (length(x), 1))

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
    @objective m Min (z' * H * z)
    @constraint m constrs G * z .<= S * x + W
    status = solve(m, suppress_warnings=true)
    if status == :Optimal
        lambdas = vec(getdual.(constrs))
        active_set = abs.(lambdas) .>= 1e-4

        G_A = G[active_set, :]
        W_A = W[active_set, :]
        S_A = S[active_set, :]
        G_I = G[.!(active_set), :]
        W_I = W[.!(active_set), :]
        S_I = S[.!(active_set), :]
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

function run_mpc(controller::PyObject, x0::AbstractVector)
    u_feedforward, x_trajectory, cost, switching_sequence = controller[:feedforward](colmat(x0))
    if isnan(u_feedforward[1][1])
        return u_feedforward[1], fill(NaN, length(u_feedforward[1]), length(x0)), x_trajectory
    end
    condensed = controller[:condense_program](switching_sequence)
    u, cost = condensed[:solve](colmat(x0))
    active_set = condensed[:get_active_set](colmat(x0), u)
    u_offset, u_linear = condensed[:get_u_sensitivity](active_set)
    vec(u_feedforward[1]), u_linear, vec.(x_trajectory)
end

include("pympc-models/models.jl")


using Base.Test

@testset "test sensitivity" begin
    mass = 1.
    l = 1.
    g = 10.
    N = 4
    A = [0. 1.;
         g/l 0.]
    B = [0 1/(mass*l^2.)]'
    Δt = .1
    pysys = PyMPC.dynamical_systems.DTLinearSystem[:from_continuous](A, B, Δt)

    x_max = [pi/6, pi/20/(N*Δt)]
    x_min = -x_max
    u_max = [mass*g*l*pi/8.]
    u_min = -u_max
    times = 0:Δt:N*Δt

    Q = 10 * eye(2)
    R = eye(1)

    X_bounds = PyMPC.geometry.Polytope[:from_bounds](reshape(x_min, 2, 1), reshape(x_max, 2, 1))[:assemble]()
    U_bounds = PyMPC.geometry.Polytope[:from_bounds](reshape(u_min, 1, 1), reshape(u_max, 1, 1))[:assemble]()
    controller = PyMPC.control.MPCController(pysys, N, "two", Q, R, X=X_bounds, U=U_bounds)

    qp = controller[:condensed_program]

    srand(1)
    for i in 1:100
        x = x_min + rand(length(x_min)) .* (x_max - x_min)
        status, u, J = PyMPC.solve_qp(qp, x)
        if status == :Optimal
            @test isapprox(u, controller[:feedforward](x)[1], atol=1e-5)
            for i in 1:length(x)
                delta = zeros(x)
                delta[i] += 1e-3
                x2 = x .+ delta
                s2, u2, J2 = PyMPC.solve_qp(qp, x2)
                @test isapprox(u2, u + J * delta, atol=1e-5)
            end
        end
    end
end


end
