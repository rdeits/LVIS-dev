@with_kw struct MIPResults
    solvetime_s::Float64
    objective_value::Float64
    objective_bound::Float64
end

struct Sample{T}
    state::Vector{T}
    input::Vector{T}
    warmstart_costs::Vector{T}
    mip::MIPResults
end

features(s::Sample) = (s.state, s.mip.objective_bound, s.mip.objective_value)

struct MPCResults{T}
    lcp_updates::Nullable{Vector{LCPSim.LCPUpdate{T, T, T}}}
    warmstart_costs::Vector{T}
    mip::MIPResults
end

function Sample(x::Union{MechanismState, LCPSim.StateRecord}, r::MPCResults)
    if isnull(r.lcp_updates)
        u = fill(NaN, num_velocities(x))
    else
        u = get(r.lcp_updates)[1].input
    end
    Sample(Vector(x), u, r.warmstart_costs, r.mip)
end


function nominal_input(x0::MechanismState{X, M}, contacts::AbstractVector{<:Point3D}=Point3D[]) where {X, M}
    # externalwrenches = BodyDict(BodyID(body) => zero(Wrench{X}) for body in bodies(mechanism))
    externalwrenches = Dict{BodyID, Wrench{X}}()
    g = x0.mechanism.gravitational_acceleration
    for point in contacts
        body = body_fixed_frame_to_body(x0.mechanism, point.frame)
        force = FreeVector3D(g.frame, -mass(x0.mechanism) / length(contacts) * g.v)
        wrench = Wrench(transform_to_root(x0, point.frame) * point, force)
        if haskey(externalwrenches, body)
            externalwrenches[BodyID(body)] += wrench
        else
            externalwrenches[BodyID(body)] = wrench
        end
    end
    v̇ = similar(velocity(x0))
    v̇ .= 0
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

function run_mpc(x0::MechanismState,
                 env::Environment,
                 params::MPCParams,
                 lqr::LQRSolution,
                 warmstart_controllers::AbstractVector{<:Function}=[])
    model = Model(solver=params.mip_solver)
    _, results_opt = LCPSim.optimize(x0, env, params.Δt, params.horizon, model)
    @objective model Min lqr_cost(results_opt, lqr)

    warmstart_costs = run_warmstarts!(model, results_opt, x0, env, params, r -> lqr_cost(r, lqr), warmstart_controllers)
    ConditionalJuMP.handle_constant_objective!(model)
    try
        solve(model, suppress_warnings=true)
    catch e
        println("captured: $e")
        return MPCResults{Float64}(nothing, nothing, warmstart_costs, mip_results)
    end

    mip_results = MIPResults(
        solvetime_s = getsolvetime(model),
        objective_value = _getvalue(getobjective(model)),
        objective_bound = getobjbound(model),
        )

    results_opt_value = getvalue.(results_opt)

    if any(isnan, results_opt_value[1].input)
        return MPCResults{Float64}(nothing, warmstart_costs, mip_results)
    else
        return MPCResults{Float64}(results_opt_value, warmstart_costs, mip_results)
    end
end

mutable struct MPCController{T, P <: MPCParams, M <: MechanismState}
    scratch_state::M
    env::Environment{T}
    params::P
    lqr::LQRSolution{T}
    warmstart_controllers::Vector{Function}
    callback::Function
end

function MPCController(model::AbstractModel,
                       params::MPCParams,
                       lqr::LQRSolution,
                       warmstart_controllers::AbstractVector{<:Function})
    scratch_state = MechanismState{Float64}(mechanism(model))
    MPCController(scratch_state,
                  environment(model),
                  params,
                  lqr,
                  convert(Vector{Function}, warmstart_controllers),
                  (state, results) -> nothing)
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
        return first(get(results.lcp_updates)).input
    else
        return zeros(length(c.lqr.u0))
    end
end
