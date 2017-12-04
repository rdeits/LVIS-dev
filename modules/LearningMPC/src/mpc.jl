
function playback(vis::Visualizer, results::AbstractVector{<:LCPUpdate}, Δt = 0.01)
    state = MechanismState{Float64}(results[1].state.mechanism)
    for result in results
        set_configuration!(state, configuration(result.state))
        settransform!(vis, state)
        for (body, contacts) in result.contacts
            for (i, contact) in enumerate(contacts)
                f = contact_force(contact)
                p = transform_to_root(state, contact.point.frame) * contact.point
                v = vis[:forces][Symbol(body)][Symbol(i)]
                setgeometry!(v, PolyLine([p.v, (p + 0.1*f).v]; end_head=ArrowHead()))
            end
        end
        sleep(Δt)
    end
end

@with_kw struct MIPResults
    solvetime_s::Float64
    objective_value::Float64
    objective_bound::Float64
end

struct Sample{T}
    state::Vector{T}
    uJ::Matrix{T}
    warmstart_costs::Vector{T}
    mip::MIPResults
end

features(s::Sample) = (s.state, s.uJ)

struct MPCResults{T}
    lcp_updates::Nullable{Vector{LCPSim.LCPUpdate{T, T, T}}}
    jacobian::Nullable{Matrix{T}}
    warmstart_costs::Vector{T}
    mip::MIPResults
end

Sample(x::Union{MechanismState, LCPSim.StateRecord}, r::MPCResults) =
    Sample(state_vector(x),
           hcat(get(r.lcp_updates)[1].input, get(r.jacobian)),
           r.warmstart_costs,
           r.mip)

struct LQRSolution{T} <: Function
    Q::Matrix{T}
    R::Matrix{T}
    K::Matrix{T}
    S::Matrix{T}
    x0::Vector{T}
    u0::Vector{T}
    Δt::T
end

function LQRSolution(x0::MechanismState{T}, Q, R, contacts::AbstractVector{<:Point3D}, Δt) where T
    u0 = nominal_input(x0, contacts)
    v0 = copy(velocity(x0))
    velocity(x0) .= 0
    RigidBodyDynamics.setdirty!(x0)
    K, S = LCPSim.ContactLQR.contact_dlqr(x0, u0, Q, R, Δt, contacts)
    set_velocity!(x0, v0)
    LQRSolution{T}(Q, R, K, S, copy(state_vector(x0)), copy(u0), Δt)
end

(c::LQRSolution)(x) = -c.K * (state_vector(x) .- c.x0) .+ c.u0

function zero_element!(sol::LQRSolution, idx::Integer)
    sol.S[idx, :] .= 0
    sol.S[:, idx] .= 0
    sol.K[:, idx] .= 0
end


@with_kw mutable struct MPCParams{S1 <: AbstractMathProgSolver, S2 <: AbstractMathProgSolver}
    Δt::Float64 = 0.05
    horizon::Int = 15
    mip_solver::S1
    lcp_solver::S2
end

function nominal_input(x0::MechanismState{X, M}, contacts::AbstractVector{<:Point3D}=Point3D[]) where {X, M}
    externalwrenches = Dict{RigidBody{M}, Wrench{X}}()
    g = x0.mechanism.gravitational_acceleration
    for point in contacts
        body = body_fixed_frame_to_body(x0.mechanism, point.frame)
        force = FreeVector3D(g.frame, -mass(x0.mechanism) / length(contacts) * g.v)
        wrench = Wrench(transform_to_root(x0, point.frame) * point, force)
        if haskey(externalwrenches, body)
            externalwrenches[body] ++ wrench
        else
            externalwrenches[body] = wrench
        end
    end
    v̇ = zeros(num_velocities(x0))
    u = inverse_dynamics(x0, v̇, externalwrenches)
    u .= clamp.(u, LCPSim.all_effort_bounds(x0.mechanism))
    u
end

function lqr_cost(results::AbstractVector{<:LCPSim.LCPUpdate},
                  lqr::LQRSolution)
    return (sum(
                (r.state.state .- lqr.x0)' * lqr.Q * (r.state.state .- lqr.x0) +
                (r.input .- lqr.u0)' * lqr.R * (r.input .- lqr.u0)
                for r in results)  +
            (results[end].state.state .- lqr.x0)' * lqr.S * (results[end].state.state .- lqr.x0))
end

joint_limit_cost(up::LCPSim.LCPUpdate) = sum([sum(jc.λ .^ 2) for jc in up.joint_contacts])

joint_limit_cost(results::AbstractVector{<:LCPSim.LCPUpdate}) =
    sum(joint_limit_cost, results)

function create_initial_state(model::Model, x0::MechanismState)
    @variable model q0[1:num_positions(x0)]
    JuMP.fix.(q0, configuration(x0))
    @variable model v0[1:num_velocities(x0)]
    JuMP.fix.(v0, velocity(x0))
    return MechanismState(x0.mechanism, q0, v0)
end

function run_warmstarts!(model::Model,
                         results::AbstractVector{<:LCPUpdate},
                         x0::MechanismState,
                         env::Environment,
                         params::MPCParams,
                         cost::Function,
                         warmstart_controllers::AbstractVector{<:Function})
    q0 = copy(configuration(x0))
    v0 = copy(velocity(x0))
    warmstarts = map(warmstart_controllers) do controller
        set_configuration!(x0, q0)
        set_velocity!(x0, v0)
        LCPSim.simulate(x0, controller, env, params.Δt, params.horizon, params.lcp_solver; relinearize=false)
    end
    warmstart_costs = [isempty(w) ? Inf : cost(w) for w in warmstarts]
    idx = indmin(warmstart_costs)
    if isfinite(warmstart_costs[idx])
        best_warmstart = warmstarts[idx]
        setvalue.(results[1:length(best_warmstart)], best_warmstart)
        ConditionalJuMP.warmstart!(model, false)
    end
    return warmstart_costs
end

function add_diagonal_cost!(model::Model, coeff=1e-6)
    # Ensure objective is strictly PD
    nvars = length(model.colCat)
    vars = [Variable(model, i) for i in 1:nvars]
    JuMP.setobjective(model, :Min, JuMP.getobjective(model) + QuadExpr(vars, vars, [1e-6 for v in vars], AffExpr([], [], 0.0)))
end


function run_mpc(x0::MechanismState,
                 env::Environment,
                 params::MPCParams,
                 lqr::LQRSolution,
                 warmstart_controllers::AbstractVector{<:Function}=[])
    # base_x = findjoint(x0.mechanism, "base_x")
    # set_configuration!(x0,
    #                    base_x,
    #                    lqr.x0[configuration_range(x0, base_x)])

    model = Model(solver=params.mip_solver)
    x0_var = create_initial_state(model, x0)
    cost = results -> (lqr_cost(results, lqr) + joint_limit_cost(results))
    _, results_opt = LCPSim.optimize(x0_var, env, params.Δt, params.horizon, model)
    @objective model Min cost(results_opt)

    warmstart_costs = run_warmstarts!(model, results_opt, x0, env, params, cost, warmstart_controllers)
    ConditionalJuMP.handle_constant_objective!(model)
    try
        status = solve(model, suppress_warnings=true)
    catch e
        println("captured: $e")
        return MPCResults{Float64}(nothing, nothing, warmstart_costs, mip_results)
    end

    mip_results = MIPResults(
        solvetime_s = getsolvetime(model),
        objective_value = _getvalue(getobjective(model)),
        objective_bound = getobjbound(model),
        )

    if any(isnan, _getvalue.(results_opt[1].input))
        return MPCResults{Float64}(nothing, nothing, warmstart_costs, mip_results)
    end

    # Now fix the binary variables and re-solve to get updated duals
    ConditionalJuMP.warmstart!(model, true)
    @assert sum(model.colCat .== :Bin) == 0 "Model should no longer have any binary variables"

    add_diagonal_cost!(model)
    try
        status = solve(model, suppress_warnings=true)
        if status != :Optimal
            return MPCResults{Float64}(getvalue.(results_opt), nothing, warmstart_costs, mip_results)
        end
    catch e
        println("captured: $e")
        return MPCResults{Float64}(getvalue.(results_opt), nothing, warmstart_costs, mip_results)
    end

    exsol = try
        ExplicitQPs.explicit_solution(model, state_vector(x0_var))
    catch e
        if isa(e, Base.LinAlg.SingularException)
            return MPCResults{Float64}(getvalue.(results_opt), nothing, warmstart_costs, mip_results)
        else
            rethrow(e)
        end
    end
    J = ExplicitQPs.jacobian(exsol, results_opt[1].input)
    return MPCResults{Float64}(getvalue.(results_opt), J, warmstart_costs, mip_results)
end

mutable struct MPCController{T, P <: MPCParams, M <: MechanismState}
    scratch_state::M
    env::Environment{T}
    params::P
    lqr::LQRSolution{T}
    warmstart_controllers::Vector{Function}
    callback::Function
end

function MPCController(mechanism::Mechanism,
                       env::Environment,
                       params::MPCParams,
                       lqr::LQRSolution,
                       warmstart_controllers::AbstractVector{<:Function})
    scratch_state = MechanismState{Float64}(mechanism)
    MPCController(scratch_state,
                  env,
                  params,
                  lqr,
                  convert(Vector{Function}, warmstart_controllers),
                  r -> nothing)
end

function (c::MPCController)(x0::Union{MechanismState, LCPSim.StateRecord})
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    results = run_mpc(c.scratch_state,
                      c.env,
                      c.params,
                      c.lqr,
                      c.warmstart_controllers)
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    c.callback(c.scratch_state, results)
    if !isnull(results.lcp_updates)
        return get(results.lcp_updates)[1].input
    else
        return zeros(length(c.lqr.u0))
    end
end
