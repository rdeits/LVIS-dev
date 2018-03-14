function run_mpc_online(x0::MechanismState,
                 env::Environment,
                 params::MPCParams,
                 lqr::LQRSolution,
                 warmstart_controllers::AbstractVector{<:Function})::Vector{Float64}
    N = params.horizon
    Δt = params.Δt
    q0 = copy(configuration(x0))
    v0 = copy(velocity(x0))
    cost = results -> (lqr_cost(results, lqr) + joint_limit_cost(results))

    warmstarts = map(warmstart_controllers) do controller
        set_configuration!(x0, q0)
        set_velocity!(x0, v0)
        LCPSim.simulate(x0, controller, env, Δt, N, params.lcp_solver)
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
        model, results_opt = LCPSim.optimize(x0, env, Δt, best_warmstart)
        setsolver(model, params.mip_solver)
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

mutable struct OnlineMPCController{T, P <: MPCParams, M <: MechanismState}
    scratch_state::M
    env::Environment{T}
    params::P
    lqr::LQRSolution{T}
    warmstart_controllers::Vector{Function}
    callback::Function
end

function OnlineMPCController(mechanism::Mechanism,
                             env::Environment,
                             params::MPCParams,
                             lqr::LQRSolution,
                             warmstart_controllers::AbstractVector{<:Function})
    scratch_state = MechanismState(mechanism)
    OnlineMPCController(scratch_state,
                  env,
                  params,
                  lqr,
                  warmstart_controllers,
                  r -> nothing)
end

function (c::OnlineMPCController)(x0::Union{MechanismState, LCPSim.StateRecord})
    set_configuration!(c.scratch_state, configuration(x0))
    set_velocity!(c.scratch_state, velocity(x0))
    run_mpc_online(c.scratch_state,
                   c.env,
                   c.params,
                   c.lqr,
                   c.warmstart_controllers)
end
