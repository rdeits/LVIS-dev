# function run_mpc_online(x0::MechanismState,
#                  env::Environment,
#                  params::BoxValkyrieMPCParams,
#                  lqr::LQRSolution,
#                  warmstart_controllers::AbstractVector{<:Function},
#                  solver=GurobiSolver(Gurobi.Env(silent=true)))::Vector{Float64}
#     N = params.horizon
#     Δt = params.Δt
#     q0 = copy(configuration(x0))
#     v0 = copy(velocity(x0))
#     cost = results -> (lqr_cost(results, lqr, Δt) + joint_limit_cost(results))

#     # Δt_sim = 0.01
#     Δt_sim = Δt
#     time_ratio = convert(Int, Δt / Δt_sim)
#     warmstarts = map(warmstart_controllers) do controller
#         set_configuration!(x0, q0)
#         set_velocity!(x0, v0)
#         LCPSim.simulate(x0, controller, env, Δt_sim, time_ratio * N, solver)
#     end

#     warmstarts = filter(x -> !isempty(x), warmstarts)
#     @show cost.(warmstarts)
#     if isempty(warmstarts)
#         warn("No feasible warmstarts this iteration")
#         return zeros(num_velocities(x0))
#     else
#         idx = indmin(cost.(warmstarts))
#         best_warmstart = warmstarts[idx]
#         set_configuration!(x0, q0)
#         set_velocity!(x0, v0)
#         model, results_opt = LCPSim.optimize(x0, env, Δt_sim, best_warmstart)
#         setsolver(model, solver)
#         @objective model Min cost(results_opt)
#         setvalue.(results_opt, best_warmstart)
#         ConditionalJuMP.warmstart!(model, true)
#         @assert sum(model.colCat .== :Bin) == 0 "Model should no longer have any binary variables"
#         status = solve(model, suppress_warnings=true)
#         if status != :Optimal
#             warn("Non-optimal status: $status")
#             return zeros(num_velocities(x0))
#         else
#             return getvalue(results_opt[1].input)
#         end
#     end
# end

# mutable struct OnlineMPCController{T, M <: MechanismState}
#     boxval::BoxValkyrie{T}
#     params::BoxValkyrieMPCParams{T}
#     scratch_state::M
#     lqr::LQRSolution{T}
#     warmstart_controllers::Vector{Function}
#     callback::Function
# end

# function OnlineMPCController(boxval::BoxValkyrie,
#                        params::BoxValkyrieMPCParams,
#                        lqr::LQRSolution,
#                        warmstart_controllers::AbstractVector{<:Function})
#     scratch_state = MechanismState(boxval.mechanism,
#                                    zeros(num_positions(boxval.mechanism)),
#                                    zeros(num_velocities(boxval.mechanism)))
#     OnlineMPCController(boxval,
#                   params,
#                   scratch_state,
#                   lqr,
#                   warmstart_controllers,
#                   r -> nothing)
# end

# function (c::OnlineMPCController)(x0::Union{MechanismState, LCPSim.StateRecord})
#     set_configuration!(c.scratch_state, configuration(x0))
#     set_velocity!(c.scratch_state, velocity(x0))
#     env = Gurobi.Env()
#     run_mpc_online(c.scratch_state,
#                    c.boxval.environment,
#                    c.params,
#                    c.lqr,
#                    c.warmstart_controllers,
#                    GurobiSolver(env,
#                                 OutputFlag=0))
# end
