__precompile__()

module CartPoles

using RigidBodyDynamics
using DrakeVisualizer
using RigidBodyTreeInspector
using LCPSim
using Polyhedra
using CDDLib
using StaticArrays: SVector
using Gurobi
using JuMP
using LearningMPC
using ExplicitQPs
using Parameters
using ConditionalJuMP

struct CartPole{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
end

function DrakeVisualizer.setgeometry!(basevis::Visualizer, cartpole::CartPole)
    vis = basevis[:robot]
    setgeometry!(vis, cartpole.mechanism, parse_urdf("cartpole.urdf", cartpole.mechanism))

    wall_radius = 1.5
    bounds = SimpleHRepresentation(vcat(eye(3), -eye(3)), vcat([wall_radius + 0.1, 0.5, 2.0], -[-wall_radius - 0.1, -0.5, -0.1]))
    for (body, contacts) in cartpole.environment.contacts
        for obstacle in contacts.obstacles
            addgeometry!(basevis[:environment], CDDPolyhedron{3, Float64}(intersect(obstacle.interior, bounds)))
        end
    end
end

function CartPole()
    mechanism = parse_urdf(Float64, "cartpole.urdf")
    world = root_body(mechanism)

    wall_radius = 1.5
    μ = 0.5
    walls = [planar_obstacle(default_frame(world), [1., 0, 0.], [-wall_radius, 0, 0.], μ), 
        planar_obstacle(default_frame(world), [-1., 0, 0.], [wall_radius, 0, 0.], μ)]
    pole = findbody(mechanism, "pole")
    env = Environment(
        Dict(pole => ContactEnvironment(
                [Point3D(default_frame(pole), SVector(0., 0, 1))],
                walls)))
    CartPole(mechanism, env)
end

@with_kw struct CartPoleMPCParams{T}
    Q::Matrix{T} = diagm([10., 100, 1, 10])
    R::Matrix{T} = diagm([0.1, 0.1])
    Δt::T = 0.01
    gap = 1e-3
    timelimit = 60
    horizon = 30
end

struct LQRSolution{T}
    Q::Matrix{T}
    R::Matrix{T}
    K::Matrix{T}
    S::Matrix{T}
    x0::Vector{T}
    u0::Vector{T}
end

function LQRSolution(x0::MechanismState, u0::AbstractVector, Q::AbstractMatrix{T}, R::AbstractMatrix{T}, contacts::AbstractVector{<:Point3D}=Point3D[]) where T
    v0 = copy(velocity(x0))
    velocity(x0) .= 0
    K, S = LCPSim.ContactLQR.contact_lqr(x0, zeros(num_velocities(x0)), Q, R, contacts)
    velocity(x0) .= v0
    LQRSolution{T}(Q, R, K, S, copy(state_vector(x0)), u0)
end

LQRController(c::LQRSolution) = x -> -c.K * (state_vector(x) .- c.x0) .+ c.u0

# (c::LQRController)(x::Union{MechanismState, LCPSim.StateRecord}) = -c.K * (state_vector(x) .- c.x0) .+ c.u0


struct MPCResults{T}
    lcp_updates::Nullable{Vector{LCPSim.LCPUpdate{T, T, T}}}
    jacobian::Nullable{Matrix{T}}
end

mutable struct MPCController{T, M <: MechanismState}
    cartpole::CartPole{T}
    params::CartPoleMPCParams{T}
    nominal_state::M
    scratch_state::M
    lqr::LQRSolution{T}
    warmstart_controllers::Vector{Function}
    callback::Function
end

function MPCController(cartpole::CartPole, 
                       params::CartPoleMPCParams, 
                       nominal_state::MechanismState,
                       lqr::LQRSolution,
                       warmstart_controllers::AbstractVector{<:Function})
    scratch_state = MechanismState(cartpole.mechanism, 
                                   zeros(num_positions(nominal_state)), 
                                   zeros(num_velocities(nominal_state)))
    MPCController(cartpole, 
                  params, 
                  nominal_state,
                  scratch_state,
                  lqr,
                  warmstart_controllers,
                  r -> nothing)
end

function (c::MPCController)(x0::Union{MechanismState, LCPSim.StateRecord})
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    results = run_mpc(c.scratch_state, 
                      c.cartpole.environment,
                      c.params, 
                      c.lqr,
                      c.nominal_state, 
                      c.warmstart_controllers,
                      GurobiSolver(OutputFlag=0, 
                                   MIPGap=c.params.gap,
                                   TimeLimit=c.params.timelimit))
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    c.callback(c.scratch_state, results)
    if !isnull(results.lcp_updates)
        return get(results.lcp_updates)[1].input
    else
        return zeros(num_velocities(c.nominal_state))
    end
end

function lqr_cost(results::AbstractVector{<:LCPSim.LCPUpdate}, lqr::LQRSolution, Δt)
    return (sum(Δt * ((r.state.state .- lqr.x0)' * lqr.Q * (r.state.state .- lqr.x0) + (r.input .- lqr.u0)' * lqr.R * (r.input .- lqr.u0)) for r in results) +
        (results[end].state.state .- lqr.x0)' * lqr.S * (results[end].state.state .- lqr.x0))
end

joint_limit_cost(up::LCPSim.LCPUpdate) = sum([sum(jc.λ .^ 2) for jc in up.joint_contacts])

function joint_limit_cost(results::AbstractVector{<:LCPSim.LCPUpdate})
    return sum(joint_limit_cost, results)
end

function create_initial_state(model::Model, x0::MechanismState)
    @variable model q0[1:num_positions(x0)]
    JuMP.fix.(q0, configuration(x0))
    @variable model v0[1:num_velocities(x0)]
    JuMP.fix.(v0, velocity(x0))
    return MechanismState(x0.mechanism, q0, v0)
end

function run_mpc(x0::MechanismState,
                 env::Environment,
                 params::CartPoleMPCParams,
                 lqr::LQRSolution,
                 x_nominal::MechanismState=x0,
                 warmstart_controllers::AbstractVector{<:Function}=[],
                 solver=GurobiSolver())
    N = params.horizon
    Δt = params.Δt
    q0 = copy(configuration(x0))
    v0 = copy(configuration(x0))
    qstar = copy(configuration(x_nominal))
    vstar = zeros(num_velocities(x_nominal))
    ustar = zeros(num_velocities(x_nominal))
    xstar = vcat(qstar, vstar)

    model = Model(solver=solver)
    x0_var = create_initial_state(model, x0)
    _, results_opt = LCPSim.optimize(x0_var, env, Δt, N, model)

    cost = results -> (lqr_cost(results, lqr, Δt) + joint_limit_cost(results))

    objective = cost(results_opt)
    @objective model Min objective

    if !isempty(warmstart_controllers)
        Δt_sim = 0.01
        time_ratio = convert(Int, Δt / Δt_sim)
        warmstarts = map(warmstart_controllers) do controller
            configuration(x0) .= q0
            velocity(x0) .= v0
            LCPSim.simulate(x0, controller, env, Δt_sim, time_ratio * N, GurobiSolver(OutputFlag=0))
        end

        warmstarts = filter(x -> !isempty(x), warmstarts)
        @show cost.(warmstarts)
        if !isempty(warmstarts)
            idx = indmin(cost.(warmstarts))
            best_warmstart = warmstarts[idx]

            setvalue.(results_opt[1:round(Int, length(best_warmstart) / time_ratio)], best_warmstart[1:time_ratio:end])
            ConditionalJuMP.warmstart!(model, false)
        end
    end

    status = solve(model, suppress_warnings=true)
    if status == :Infeasible
        return MPCResults{Float64}(nothing, nothing)
    end
    
    # Now fix the binary variables and re-solve to get updated duals
    ConditionalJuMP.warmstart!(model, true)
    @assert sum(model.colCat .== :Bin) == 0 "Model should no longer have any binary variables"
    
    # Ensure objective is strictly PD
    nvars = length(model.colCat)
    vars = [Variable(model, i) for i in 1:nvars]
    @objective model Min objective + QuadExpr(vars, vars, [1e-6 for v in vars], AffExpr([], [], 0.0))
    status = solve(model, suppress_warnings=true)
    if status != :Optimal
        return MPCResults{Float64}(getvalue.(results_opt), nothing)
    end
    exsol = try
        ExplicitQPs.explicit_solution(model, state_vector(x0_var))
    catch e
        if isa(e, Base.LinAlg.SingularException)
            return MPCResults{Float64}(getvalue.(results_opt), nothing)
        else
            rethrow(e)
        end
    end
    J = ExplicitQPs.jacobian(exsol, results_opt[1].input)
    return MPCResults{Float64}(getvalue.(results_opt), J)
end

function run_mpc_online(x0::MechanismState,
                 env::Environment,
                 params::CartPoleMPCParams,
                 lqr::LQRSolution,
                 x_nominal::MechanismState,
                 warmstart_controllers::AbstractVector{<:Function},
                 solver=GurobiSolver(Gurobi.Env()))::Vector{Float64}
    N = params.horizon
    Δt = params.Δt
    q0 = copy(configuration(x0))
    v0 = copy(velocity(x0))
    cost = results -> (lqr_cost(results, lqr, Δt) + joint_limit_cost(results))

    # Δt_sim = 0.01
    Δt_sim = Δt
    time_ratio = convert(Int, Δt / Δt_sim)
    warmstarts = map(warmstart_controllers) do controller
        configuration(x0) .= q0
        velocity(x0) .= v0
        LCPSim.simulate(x0, controller, env, Δt_sim, time_ratio * N, solver)
    end

    warmstarts = filter(x -> !isempty(x), warmstarts)
    @show cost.(warmstarts)
    if isempty(warmstarts)
        warn("No feasible warmstarts this iteration")
        return zeros(num_velocities(x0))
    else
        idx = indmin(cost.(warmstarts))
        best_warmstart = warmstarts[idx]
        configuration(x0) .= q0
        velocity(x0) .= v0
        model, results_opt = LCPSim.optimize(x0, env, Δt_sim, best_warmstart)
        setsolver(model, solver)
        @objective model Min cost(results_opt)
        setvalue.(results_opt, best_warmstart)
        ConditionalJuMP.warmstart!(model, true)
        @assert sum(model.colCat .== :Bin) == 0 "Model should no longer have any binary variables"
        status = solve(model, suppress_warnings=true)
        if status != :Optimal
            warn("Non-optimal status: $status")
            return zeros(num_velocities(x0))
        else
            return getvalue(results_opt[1].input)
        end
    end
end

mutable struct OnlineMPCController{T, M <: MechanismState}
    cartpole::CartPole{T}
    params::CartPoleMPCParams{T}
    nominal_state::M
    scratch_state::M
    lqr::LQRSolution{T}
    warmstart_controllers::Vector{Function}
    callback::Function
end

function OnlineMPCController(cartpole::CartPole, 
                       params::CartPoleMPCParams, 
                       nominal_state::MechanismState,
                       lqr::LQRSolution,
                       warmstart_controllers::AbstractVector{<:Function})
    scratch_state = MechanismState(cartpole.mechanism, 
                                   zeros(num_positions(nominal_state)), 
                                   zeros(num_velocities(nominal_state)))
    OnlineMPCController(cartpole, 
                  params, 
                  nominal_state,
                  scratch_state,
                  lqr,
                  warmstart_controllers,
                  r -> nothing)
end

function (c::OnlineMPCController)(x0::Union{MechanismState, LCPSim.StateRecord})
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    env = Gurobi.Env()
    run_mpc_online(c.scratch_state, 
                   c.cartpole.environment,
                   c.params, 
                   c.lqr,
                   c.nominal_state, 
                   c.warmstart_controllers,
                   GurobiSolver(env,
                                OutputFlag=0))
end

end
