__precompile__()

module BoxValkyries

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

struct BoxValkyrie{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
end

const urdf = joinpath(Pkg.dir("LCPSim"), "examples", "box_valkyrie.urdf")

function DrakeVisualizer.setgeometry!(basevis::Visualizer, boxval::BoxValkyrie)
    vis = basevis[:robot]
    setgeometry!(vis, boxval.mechanism, parse_urdf(urdf, boxval.mechanism))

    wall_radius = 1.5
    bounds = SimpleHRepresentation(vcat(eye(3), -eye(3)), vcat([wall_radius + 0.1, 0.5, 2.0], -[-wall_radius - 0.1, -0.5, -0.1]))
    for (body, contacts) in boxval.environment.contacts
        for obstacle in contacts.obstacles
            addgeometry!(basevis[:environment], CDDPolyhedron{3, Float64}(intersect(obstacle.interior, bounds)))
        end
    end
end

function BoxValkyrie()
    urdf_mech = parse_urdf(Float64, urdf)
    mechanism, base = planar_revolute_base()
    attach!(mechanism, base, urdf_mech)
    world = root_body(mechanism)


    floor = planar_obstacle(default_frame(world), [0, 0, 1.], [0, 0, 0.], 1.)
    wall = planar_obstacle(default_frame(world), [1., 0, 0], [-0.7, 0, 0], 1.)

    contact_limbs = findbody.(mechanism, ["rh", "lh", "rf", "lf"])
    hands = findbody.(mechanism, ["rh", "lh"])
    feet = findbody.(mechanism, ["rf", "lf"])

    env = Environment(
        Dict(vcat(
                [body => ContactEnvironment(
                    [Point3D(default_frame(body), SVector(0., 0, 0))],
                    [floor])
                    for body in feet],
                [body => ContactEnvironment(
                    [Point3D(default_frame(body), SVector(0., 0, 0))],
                    [wall])
                    for body in [findbody(mechanism, "lh")]]
                )));

    BoxValkyrie(mechanism, env)
end

@with_kw struct BoxValkyrieMPCParams{T}
    Q::Matrix{T} = diagm([
        0.01,
        2,
        100,
        0.1, # rhx
        0.1, # lhx
        1,   # rfx
        1,   # lfx
        0.1, # rhz
        0.1, # lhz
        0.1,   # rfz
        0.1,   # lfz
        10,
        1,
        100,
        fill(0.01, 8)...
        ])
    R::Matrix{T} = diagm(fill(0.0005, 11))
    Δt::T = 0.04
    gap = 1e-2
    timelimit = 60
    horizon = 15
end

function nominal_input(val::BoxValkyrie, x0::MechanismState)
    mechanism = val.mechanism
    u_nominal = clamp.(inverse_dynamics(x0, zeros(num_velocities(x0))), LCPSim.all_effort_bounds(x0.mechanism))
    feet = findbody.(x0.mechanism, ["rf", "lf"])
    weight = mass(x0.mechanism) * mechanism.gravitational_acceleration.v[3]
    u_nominal[parentindexes(velocity(x0, findjoint(x0.mechanism, "core_to_rf_z")))...] += weight / 2
    u_nominal[parentindexes(velocity(x0, findjoint(x0.mechanism, "core_to_lf_z")))...] += weight / 2
    u_nominal
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
    RigidBodyDynamics.setdirty!(x0)
    K, S = LCPSim.ContactLQR.contact_lqr(x0, zeros(num_velocities(x0)), Q, R, contacts)
    set_velocity!(x0, v0)
    LQRSolution{T}(Q, R, K, S, copy(state_vector(x0)), u0)
end

LQRController(c::LQRSolution) = x -> -c.K * (state_vector(x) .- c.x0) .+ c.u0

# (c::LQRController)(x::Union{MechanismState, LCPSim.StateRecord}) = -c.K * (state_vector(x) .- c.x0) .+ c.u0


struct MPCResults{T}
    lcp_updates::Nullable{Vector{LCPSim.LCPUpdate{T, T, T}}}
    jacobian::Nullable{Matrix{T}}
end

mutable struct MPCController{T, M <: MechanismState}
    boxval::BoxValkyrie{T}
    params::BoxValkyrieMPCParams{T}
    scratch_state::M
    lqr::LQRSolution{T}
    warmstart_controllers::Vector{Function}
    callback::Function
end

function MPCController(boxval::BoxValkyrie,
                       params::BoxValkyrieMPCParams,
                       lqr::LQRSolution,
                       warmstart_controllers::AbstractVector{<:Function})
    scratch_state = MechanismState(boxval.mechanism,
                                   zeros(num_positions(boxval.mechanism)),
                                   zeros(num_velocities(boxval.mechanism)))
    MPCController(boxval,
                  params,
                  scratch_state,
                  lqr,
                  warmstart_controllers,
                  r -> nothing)
end

function (c::MPCController)(x0::Union{MechanismState, LCPSim.StateRecord})
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    env = Gurobi.Env()
    solver = GurobiSolver(env,
                          OutputFlag=1,
                          FeasibilityTol=1e-4,
                          MIPGap=c.params.gap,
                          TimeLimit=c.params.timelimit)
    results = run_mpc(c.boxval,
                      c.scratch_state,
                      c.boxval.environment,
                      c.params,
                      c.lqr,
                      c.warmstart_controllers,
                      solver)
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    c.callback(c.scratch_state, results)
    if !isnull(results.lcp_updates)
        return get(results.lcp_updates)[1].input
    else
        return zeros(length(c.lqr.u0))
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

function _run_optimization(boxval, x0, env, Δt, N; x_nominal=x0, solver=GurobiSolver())
    qstar = copy(configuration(x_nominal))
    vstar = zeros(num_velocities(x_nominal))
    ustar = nominal_input(boxval, x_nominal)
    feet = findbody.(x0.mechanism, ["rf", "lf"])
    
    current_feet_positions = [transform_to_root(x0, default_frame(foot)) * Point3D(default_frame(foot), SVector(0., 0, 0)) for foot in feet]
    qstar[1] = mean([p.v[1] for p in current_feet_positions])
        
    model, results_opt = LCPSim.optimize(x0, env, Δt, N, Model(solver=solver))

    qqf = zeros(num_positions(x0))
    qqf[configuration_range(x0, findjoint(x0.mechanism, "base_x"))]        .= 0
    qqf[configuration_range(x0, findjoint(x0.mechanism, "base_z"))]        .= 100
    qqf[configuration_range(x0, findjoint(x0.mechanism, "base_rotation"))] .= 10
    qqf[configuration_range(x0, findjoint(x0.mechanism, "core_to_rh_x"))]  .= 1
    qqf[configuration_range(x0, findjoint(x0.mechanism, "core_to_lh_x"))]  .= 1
    qqf[configuration_range(x0, findjoint(x0.mechanism, "core_to_rf_x"))]  .= 10
    qqf[configuration_range(x0, findjoint(x0.mechanism, "core_to_lf_x"))]  .= 10
    qqf[configuration_range(x0, findjoint(x0.mechanism, "core_to_rh_z"))]  .= 1
    qqf[configuration_range(x0, findjoint(x0.mechanism, "core_to_lh_z"))]  .= 1
    qqf[configuration_range(x0, findjoint(x0.mechanism, "core_to_rf_z"))]  .= 1
    qqf[configuration_range(x0, findjoint(x0.mechanism, "core_to_lf_z"))]  .= 1
    qvf = zeros(num_positions(x0))
    Qf = diagm(vcat(qqf, qvf))

    w = 0.1 * ones(num_positions(x0))
    w[1] = 0
    w[2] = 10
    w[3] = 1
    w[6:7] = 1

    Δxf = vcat(configuration(results_opt[end].state) .- qstar,
               velocity(results_opt[end].state) .- vstar)
    
    objective = (
        Δxf' * Qf * Δxf
        # + 10 * sum(w .* (qstar .- configuration(results_opt[end].state)).^2)
        )
    
    qq = zeros(num_positions(x0))
    qq[configuration_range(x0, findjoint(x0.mechanism, "base_x"))]        .= 0
    qq[configuration_range(x0, findjoint(x0.mechanism, "base_z"))]        .= 0.1
    qq[configuration_range(x0, findjoint(x0.mechanism, "base_rotation"))] .= 0.1
    qq[configuration_range(x0, findjoint(x0.mechanism, "core_to_rh_x"))]  .= 0.01
    qq[configuration_range(x0, findjoint(x0.mechanism, "core_to_lh_x"))]  .= 0.01
    qq[configuration_range(x0, findjoint(x0.mechanism, "core_to_rf_x"))]  .= 0.01
    qq[configuration_range(x0, findjoint(x0.mechanism, "core_to_lf_x"))]  .= 0.01
    qq[configuration_range(x0, findjoint(x0.mechanism, "core_to_rh_z"))]  .= 0.01
    qq[configuration_range(x0, findjoint(x0.mechanism, "core_to_lh_z"))]  .= 0.01
    qq[configuration_range(x0, findjoint(x0.mechanism, "core_to_rf_z"))]  .= 0.01
    qq[configuration_range(x0, findjoint(x0.mechanism, "core_to_lf_z"))]  .= 0.01

    qv = zeros(num_velocities(x0))
    qv[velocity_range(x0, findjoint(x0.mechanism, "base_x"))] .= 0.1

    Q = diagm(vcat(qq, qv))
    R = diagm(fill(0.0001, num_velocities(x0)))

    w[6:7] = 0.1
    w[2] = 1
    w .*= 0.1
    @show w
    @show diag(Q)
    
    for r in results_opt
        Δx = vcat(configuration(r.state) .- qstar,
                  velocity(r.state) .- vstar)
        objective += (
            0
            + r.input' * R * r.input
            # + 0.0001 * sum((r.input).^2)
            + Δx' * Q * Δx
            # + sum(w .* (qstar .- configuration(r.state)).^2)
            # + 0.1 * velocity(r.state)[1]^2
            )
            
    end

    objective += joint_limit_cost(results_opt)
    
    @objective model Min objective

    
    contacts = [Point3D(default_frame(body), SVector(0., 0, 0)) for body in feet]
    # qq = [100, 0.1, 1, fill(0.1, num_positions(x0) - 3)...]
    # Q = diagm(vcat(qq, fill(0.1, num_velocities(x0))))
    # R = 0.001 * eye(num_velocities(x0))
    # K, S = LCPSim.ContactLQR.contact_lqr(MechanismState(x0.mechanism, qstar, vstar), 
    #         ustar, Q, R, contacts)
    qq = [100, 0.1, 10, fill(0.1, num_positions(x0) - 3)...]
    Q = diagm(vcat(qq, fill(0.1, num_velocities(x0))))
    R = 0.01 * eye(num_velocities(x0))
    Δt_sim = Δt
    K, S = LCPSim.ContactLQR.contact_dlqr(MechanismState(x0.mechanism, qstar, vstar), ustar, Q, R, contacts, Δt_sim)

    controller = x -> begin
        -K * (state_vector(x) - vcat(qstar, vstar)) .+ ustar
    end
    time_ratio = convert(Int, Δt / Δt_sim)
    results = LCPSim.simulate(x0, controller, env, Δt_sim, time_ratio * N, GurobiSolver(Gurobi.Env(), OutputFlag=0))
    setvalue.(results_opt, results[1:time_ratio:end])
    ConditionalJuMP.warmstart!(model, false)
    
    solve(model)
    getvalue.(results_opt)
end

function run_mpc(boxval::BoxValkyrie,
                 x0::MechanismState,
                 env::Environment,
                 params::BoxValkyrieMPCParams,
                 lqr::LQRSolution,
                 warmstart_controllers::AbstractVector{<:Function}=[],
                 solver=GurobiSolver(Gurobi.Env(), OutputFlag=0))
    x_nominal = MechanismState(x0.mechanism, lqr.x0[1:num_positions(x0)], lqr.x0[num_positions(x0) + 1:end])
    results = _run_optimization(boxval, x0, env, params.Δt, params.horizon,
                             x_nominal=x_nominal,
                             solver=solver)
    return MPCResults{Float64}(results, nothing)


    N = params.horizon
    Δt = params.Δt
    q0 = copy(configuration(x0))
    v0 = copy(velocity(x0))

    model = Model(solver=solver)
    x0_var = create_initial_state(model, x0)
    cost = results -> (lqr_cost(results, lqr, Δt) + joint_limit_cost(results))


    has_warmstart = false
    if !isempty(warmstart_controllers)
        warmstarts = map(warmstart_controllers) do controller
            set_configuration!(x0, q0)
            set_velocity!(x0, v0)
            LCPSim.simulate(x0, controller, env, Δt, N, GurobiSolver(Gurobi.Env(), OutputFlag=0))
        end

        warmstarts = filter(x -> !isempty(x), warmstarts)
        @show cost.(warmstarts)
        if !isempty(warmstarts)
            idx = indmin(cost.(warmstarts))
            best_warmstart = warmstarts[idx]

            if length(best_warmstart) == N
                _, results_opt = LCPSim.optimize(x0_var, env, Δt, best_warmstart, model)
                has_warmstart = true
            end
        end
    end
    @show has_warmstart

    if !has_warmstart
        _, results_opt = LCPSim.optimize(x0_var, env, Δt, N, model)
    end

    objective = cost(results_opt)
    @objective model Min objective

    status = solve(model, suppress_warnings=true)
    @show status
    if any(isnan, JuMP.getvalue(results_opt[1].input))
        return MPCResults{Float64}(nothing, nothing)
    end

    # Now fix the binary variables and re-solve to get updated duals
    ConditionalJuMP.warmstart!(model, true)
    for i in 1:length(model.colCat)
      if model.colCat[i] == :Bin
        @show Variable(model, i)
      end
    end
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
                 params::BoxValkyrieMPCParams,
                 lqr::LQRSolution,
                 warmstart_controllers::AbstractVector{<:Function},
                 solver=GurobiSolver(Gurobi.Env(silent=true)))::Vector{Float64}
    N = params.horizon
    Δt = params.Δt
    q0 = copy(configuration(x0))
    v0 = copy(velocity(x0))
    cost = results -> (lqr_cost(results, lqr, Δt) + joint_limit_cost(results))

    # Δt_sim = 0.01
    Δt_sim = Δt
    time_ratio = convert(Int, Δt / Δt_sim)
    warmstarts = map(warmstart_controllers) do controller
        set_configuration!(x0, q0)
        set_velocity!(x0, v0)
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
        set_configuration!(x0, q0)
        set_velocity!(x0, v0)
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
    boxval::BoxValkyrie{T}
    params::BoxValkyrieMPCParams{T}
    scratch_state::M
    lqr::LQRSolution{T}
    warmstart_controllers::Vector{Function}
    callback::Function
end

function OnlineMPCController(boxval::BoxValkyrie,
                       params::BoxValkyrieMPCParams,
                       lqr::LQRSolution,
                       warmstart_controllers::AbstractVector{<:Function})
    scratch_state = MechanismState(boxval.mechanism,
                                   zeros(num_positions(boxval.mechanism)),
                                   zeros(num_velocities(boxval.mechanism)))
    OnlineMPCController(boxval,
                  params,
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
                   c.boxval.environment,
                   c.params,
                   c.lqr,
                   c.warmstart_controllers,
                   GurobiSolver(env,
                                OutputFlag=0))
end

end
