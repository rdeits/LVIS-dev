const hopper_urdf = joinpath(@__DIR__, "urdf", "hopper.urdf")

struct Hopper{T} <: AbstractModel{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
    foot::RigidBody{T}
end

mechanism(h::Hopper) = h.mechanism
environment(h::Hopper) = h.environment
urdf(::Hopper) = hopper_urdf

function Hopper()
    mechanism = parse_urdf(Float64, hopper_urdf)
    env = LCPSim.parse_contacts(mechanism, hopper_urdf, 1.0, :xz)
    foot = findbody(mechanism, "foot")
    Hopper(mechanism, env, foot)
end

function nominal_state(c::Hopper)
    x = MechanismState{Float64}(c.mechanism)
    set_configuration!(x, findjoint(c.mechanism, "base_z"), [2.0])
    set_configuration!(x, findjoint(c.mechanism, "foot_extension"), [0.75])
    x
end

function default_costs(c::Hopper)
    Q = diagm([10., 1e-1, 1e-3, 1])
    R = diagm([1e-3, 1e-3])
    Q, R
end

function LearningMPC.MPCParams(c::Hopper)
    mpc_params = LearningMPC.MPCParams(
        Δt=0.05,
        horizon=10,
        mip_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0,
            TimeLimit=5,
            MIPGap=1e-1,
            FeasibilityTol=1e-3),
        lcp_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0))
end

function LearningMPC.LQRSolution(c::Hopper, params::MPCParams=MPCParams(c))
    xstar = nominal_state(c)
    Q, R = default_costs(c)
    lqrsol = LearningMPC.LQRSolution(xstar, Q, R, params.Δt, [Point3D(default_frame(c.foot), 0., 0., 0.)])
    lqrsol.S .= 1 ./ params.Δt .* Q
    lqrsol
end
