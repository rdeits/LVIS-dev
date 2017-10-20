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

struct MPCResults{T}
    lcp_updates::Nullable{Vector{LCPSim.LCPUpdate{T, T, T}}}
    jacobian::Nullable{Matrix{T}}
end

mutable struct MPCController{T, M <: MechanismState}
    cartpole::CartPole{T}
    params::CartPoleMPCParams{T}
    nominal_state::M
    scratch_state::M
    callback::Function
end

function MPCController(cartpole::CartPole, 
                       params::CartPoleMPCParams, 
                       nominal_state::MechanismState)
    scratch_state = MechanismState(cartpole.mechanism, 
                                   zeros(num_positions(nominal_state)), 
                                   zeros(num_velocities(nominal_state)))
    MPCController(cartpole, 
                  params, 
                  nominal_state,
                  scratch_state,
                  r -> nothing)
end

function (c::MPCController)(x0::Union{MechanismState, LCPSim.StateRecord})
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    results = run_mpc(c.scratch_state, 
                      c.cartpole.environment,
                      c.params, 
                      c.nominal_state, 
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

function run_mpc(x0::MechanismState,
                 env::Environment,
                 params::CartPoleMPCParams,
                 x_nominal=x0,
                 solver=GurobiSolver())
    N = params.horizon
    Δt = params.Δt
    qstar = copy(configuration(x_nominal))
    vstar = zeros(num_velocities(x_nominal))
    ustar = zeros(num_velocities(x_nominal))

    contacts = Point3D[]
    v0 = copy(velocity(x0))
    set_velocity!(x0, zeros(num_velocities(x0)))
    K, S = LCPSim.ContactLQR.contact_lqr(x0, zeros(num_velocities(x0)), params.Q, params.R, contacts)
    set_velocity!(x0, v0)
    
    model = Model(solver=solver)
    @variable model q0[1:num_positions(x0)]
    JuMP.fix.(q0, configuration(x0))
    @variable model v0[1:num_velocities(x0)]
    JuMP.fix.(v0, velocity(x0))
    
    _, results_opt = LCPSim.optimize(MechanismState(x0.mechanism, q0, v0), env, Δt, N, model)
    
    objective = (
        sum(Δt * (r.state.state' * params.Q * r.state.state + r.input' * params.R * r.input) for r in results_opt)
       + (results_opt[end].state.state' * S * results_opt[end].state.state)
        )

    for r in results_opt
        for (joint, jrs) in r.joint_contacts
            for joint_result in jrs
                objective += joint_result.λ^2
            end
        end 
    end
    
    @objective model Min objective


    controller = x -> begin
        -K * (state_vector(x) - vcat(qstar, vstar)) .+ ustar
    end
    
    Δt_sim = 0.01
    time_ratio = convert(Int, Δt / Δt_sim)
    results_lqr = LCPSim.simulate(x0, controller, env, Δt_sim, time_ratio * N, GurobiSolver(OutputFlag=0))
    if length(results_lqr[1:time_ratio:end]) == length(results_opt)
        setvalue.(results_opt, results_lqr[1:time_ratio:end])
        ConditionalJuMP.warmstart!(model, false)
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
    status = solve(model)
    if status != :Optimal
        return MPCResults{Float64}(getvalue.(results_opt), nothing)
    end
    exsol = try
        ExplicitQPs.explicit_solution(model, vcat(q0, v0))
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


end
