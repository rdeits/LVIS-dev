const slider_urdf = joinpath(@__DIR__, "urdf", "slider.urdf")

struct Slider{T} <: AbstractModel{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
end

mechanism(c::Slider) = c.mechanism
environment(c::Slider) = c.environment
urdf(c::Slider) = slider_urdf 

function Slider()
    mechanism = parse_urdf(Float64, slider_urdf)
    env = LCPSim.parse_contacts(mechanism, slider_urdf, 1.0, :xz)
    Slider(mechanism, env)
end

function nominal_state(c::Slider)
    x = MechanismState{Float64}(c.mechanism)
end

function default_costs(c::Slider)
    Q = diagm([1., 1])
    R = diagm([0.1])
    Q, R
end

function LearningMPC.MPCParams(c::Slider)
    mpc_params = LearningMPC.MPCParams(
        Δt=0.05,
        horizon=20,
        mip_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0,
            TimeLimit=5,
            MIPGap=1e-1,
            FeasibilityTol=1e-3),
        lcp_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0))
end

function LearningMPC.LQRSolution(c::Slider, params::MPCParams=MPCParams(c))
    xstar = nominal_state(c)
    Q, R = default_costs(c)
    lqrsol = LearningMPC.LQRSolution(xstar, Q, R, params.Δt, Point3D[])
end
