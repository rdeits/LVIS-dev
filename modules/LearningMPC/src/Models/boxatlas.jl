box_atlas_urdf = joinpath(@__DIR__, "urdf", "box_atlas.urdf")

struct BoxAtlas{T} <: AbstractModel{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
    floating_base::Joint{T}
    feet::Dict{Symbol, RigidBody{T}}
    hands::Dict{Symbol, RigidBody{T}}
end

mechanism(b::BoxAtlas) = b.mechanism
environment(b::BoxAtlas) = b.environment
urdf(b::BoxAtlas) = box_atlas_urdf

function BoxAtlas()
    mechanism = parse_urdf(Float64, box_atlas_urdf)
    floating_base = findjoint(mechanism, "floating_base")
    floating_base.position_bounds .= RigidBodyDynamics.Bounds(-10, 10)
    floating_base.velocity_bounds .= RigidBodyDynamics.Bounds(-1000, 1000)
    floating_base.effort_bounds .= RigidBodyDynamics.Bounds(0, 0)
    env = LCPSim.parse_contacts(mechanism, box_atlas_urdf, 1.0, :xz)
    feet = Dict(:left => findbody(mechanism, "lf"),
                :right => findbody(mechanism, "rf"))
    hands = Dict(:left => findbody(mechanism, "lh"),
                 :right => findbody(mechanism, "rh"))
    floor = findbody(mechanism, "floor")
    wall = findbody(mechanism, "wall")
    LCPSim.filter_contacts!(env, mechanism, 
        Dict(hands[:right] => [],
             hands[:left] => [wall],
             feet[:right] => [floor],
             feet[:left] => [floor, wall]))
    BoxAtlas(mechanism, env, floating_base, feet, hands)
end

function nominal_state(robot::BoxAtlas)
    m = mechanism(robot)
    xstar = MechanismState{Float64}(m)
    set_configuration!(xstar, findjoint(m, "floating_base"), [0, 1.05, 0])
    set_configuration!(xstar, findjoint(m, "core_to_lf_extension"), [0.8])
    set_configuration!(xstar, findjoint(m, "core_to_rf_extension"), [0.8])
    set_configuration!(xstar, findjoint(m, "core_to_lh_extension"), [0.5])
    set_configuration!(xstar, findjoint(m, "core_to_rh_extension"), [0.5])
    xstar
end

function default_costs(robot::BoxAtlas, r=0.01)
    x = nominal_state(robot)

    qq = zeros(num_positions(x))
    qq[configuration_range(x, findjoint(x.mechanism, "floating_base"))] = [0, 10, 800]
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rh_extension"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lh_extension"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rh_rotation"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lh_rotation"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rf_extension"))]  .= 0.1
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lf_extension"))]  .= 0.1
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rf_rotation"))]  .= 0.1
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lf_rotation"))]  .= 0.1

    qv = fill(0.01, num_velocities(x))
    qv[velocity_range(x, findjoint(x.mechanism, "floating_base"))] = [10, 1, 50]

    Q = diagm(vcat(qq, qv))

    # minimize (rx - lx)^2 = rx^2 - 2rxlx + lx^2
    rx = configuration_range(x, findjoint(x.mechanism, "core_to_rf_extension"))
    lx = configuration_range(x, findjoint(x.mechanism, "core_to_lf_extension"))
    w_centering = 1
    Q[rx, rx] += w_centering
    Q[lx, lx] += w_centering
    Q[lx, rx] -= w_centering
    Q[rx, lx] -= w_centering
    rθ = configuration_range(x, findjoint(x.mechanism, "core_to_rf_rotation"))
    lθ = configuration_range(x, findjoint(x.mechanism, "core_to_lf_rotation"))
    w_centering = 1
    Q[rθ, rθ] += w_centering
    Q[lθ, lθ] += w_centering
    Q[lθ, rθ] -= w_centering
    Q[rθ, lθ] -= w_centering

    rr = fill(r, num_velocities(x))
    R = diagm(rr)
    Q, R
end

function LearningMPC.MPCParams(robot::BoxAtlas)
    mpc_params = LearningMPC.MPCParams(
        Δt=0.05,
        horizon=10,
        mip_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0,
            TimeLimit=5,
            MIPGap=1e-1,
            FeasibilityTol=1e-3),
        lcp_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0))
end

function LearningMPC.LQRSolution(robot::BoxAtlas, params::MPCParams=MPCParams(robot), zero_base_x=true)
    xstar = nominal_state(robot)
    Q, R = default_costs(robot)
    lqrsol = LearningMPC.LQRSolution(xstar, Q, R, params.Δt, 
        [Point3D(default_frame(robot.feet[:left]), 0., 0., 0.),
         Point3D(default_frame(robot.feet[:right]), 0., 0., 0.)])
    if zero_base_x
        lqrsol.S[1,:] .= 0
        lqrsol.S[:,1] .= 0
        lqrsol.K[:,1] .= 0
    end
    lqrsol
end

