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

const urdf = joinpath(@__DIR__, "box_valkyrie.urdf")

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
    wall = planar_obstacle(default_frame(world), [1., 0, 0], [-1.0, 0, 0], 1.)

    # contact_limbs = findbody.(mechanism, ["rh", "lh", "rf", "lf"])
    # hands = findbody.(mechanism, ["rh", "lh"])
    # feet = findbody.(mechanism, ["rf", "lf"])
    rf = findbody(mechanism, "rf")
    lf = findbody(mechanism, "lf")
    rh = findbody(mechanism, "rh")
    lh = findbody(mechanism, "lh")

    env = Environment(
        Dict(
             rf => ContactEnvironment(
                [Point3D(default_frame(rf), SVector(0., 0, 0))],
                [floor]),
             lf => ContactEnvironment(
                [Point3D(default_frame(lf), SVector(0., 0, 0))],
                [floor, wall]),
             rh => ContactEnvironment(
                [Point3D(default_frame(rh), SVector(0., 0, 0))],
                [floor]),
             lh => ContactEnvironment(
                [Point3D(default_frame(lh), SVector(0., 0, 0))],
                [floor, wall]),
             ))

    BoxValkyrie(mechanism, env)
end

function nominal_state(boxval::BoxValkyrie)
    mechanism = boxval.mechanism
    xstar = MechanismState{Float64}(mechanism)
    set_configuration!(xstar, findjoint(mechanism, "base_z"), [1.05])
    set_configuration!(xstar, findjoint(mechanism, "core_to_lf_extension"), [0.8])
    set_configuration!(xstar, findjoint(mechanism, "core_to_rf_extension"), [0.8])
    set_configuration!(xstar, findjoint(mechanism, "core_to_lh_extension"), [0.5])
    set_configuration!(xstar, findjoint(mechanism, "core_to_rh_extension"), [0.5])
    xstar
end

function default_costs(x::MechanismState)
    qq = zeros(num_positions(x))
    qq[configuration_range(x, findjoint(x.mechanism, "base_x"))]        .= 0
    qq[configuration_range(x, findjoint(x.mechanism, "base_z"))]        .= 10
    qq[configuration_range(x, findjoint(x.mechanism, "base_rotation"))] .= 500
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rh_extension"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lh_extension"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rh_rotation"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lh_rotation"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rf_extension"))]  .= 0.1
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lf_extension"))]  .= 0.1
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rf_rotation"))]  .= 0.01
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lf_rotation"))]  .= 0.01

    qv = fill(1e-4, num_velocities(x))
    # qv[velocity_range(x, findjoint(x.mechanism, "base_x"))] .= 0.1

    Q = diagm(vcat(qq, qv))
    # # minimize (rx - lx)^2 = rx^2 - 2rxlx + lx^2
    rx = configuration_range(x, findjoint(x.mechanism, "core_to_rf_extension"))
    lx = configuration_range(x, findjoint(x.mechanism, "core_to_lf_extension"))
    w_centering = 10
    Q[rx, rx] += w_centering
    Q[lx, lx] += w_centering
    Q[lx, rx] -= w_centering
    Q[rx, lx] -= w_centering
    rθ = configuration_range(x, findjoint(x.mechanism, "core_to_rf_rotation"))
    lθ = configuration_range(x, findjoint(x.mechanism, "core_to_lf_rotation"))
    w_centering = 10
    Q[rθ, rθ] += w_centering
    Q[lθ, lθ] += w_centering
    Q[lθ, rθ] -= w_centering
    Q[rθ, lθ] -= w_centering

    rr = fill(0.002, num_velocities(x))
    rr[velocity_range(x, findjoint(x.mechanism, "core_to_rf_extension"))] .= 0.01
    rr[velocity_range(x, findjoint(x.mechanism, "core_to_lf_extension"))] .= 0.01
    rr[velocity_range(x, findjoint(x.mechanism, "core_to_rf_rotation"))] .= 0.01
    rr[velocity_range(x, findjoint(x.mechanism, "core_to_lf_rotation"))] .= 0.01
    R = diagm(rr)
    Q, R
end

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

end
