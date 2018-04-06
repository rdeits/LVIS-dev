const cartpole_urdf = joinpath(@__DIR__, "urdf", "cartpole_with_walls.urdf")

struct CartPole{T} <: AbstractModel{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
end

mechanism(c::CartPole) = c.mechanism
environment(c::CartPole) = c.environment
urdf(c::CartPole) = cartpole_urdf

function CartPole()
    mechanism = parse_urdf(Float64, cartpole_urdf)
    env = LCPSim.parse_contacts(mechanism, cartpole_urdf, 0.5, :xz)
    CartPole(mechanism, env)
end

function nominal_state(c::CartPole)
    x = MechanismState{Float64}(c.mechanism)
end

function default_costs(c::CartPole)
    Q = diagm([10., 100, 1, 10])
    R = diagm([0.1, 0.1])
    Q, R
end

function LearningMPC.MPCParams(c::CartPole)
    mpc_params = LearningMPC.MPCParams(
        Δt=0.025,
        horizon=20,
        mip_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0,
            TimeLimit=5,
            MIPGap=1e-1,
            FeasibilityTol=1e-3),
        lcp_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0))
end

function LearningMPC.LQRSolution(c::CartPole, params::MPCParams=MPCParams(c))
    xstar = nominal_state(c)
    Q, R = default_costs(c)
    lqrsol = LearningMPC.LQRSolution(xstar, Q, R, params.Δt, Point3D[])
end
