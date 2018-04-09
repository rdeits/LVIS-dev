__precompile__()

module LearningMPC

using LCPSim
using LCPSim: LCPUpdate, contact_force, _getvalue
using RigidBodyDynamics
using MeshCatMechanisms: MechanismVisualizer, animate
using Parameters: @with_kw
using MathProgBase.SolverInterface: AbstractMathProgSolver
using JuMP
using CoordinateTransformations: AffineMap
using Flux
using FluxExtensions
import ConditionalJuMP

export playback,
       MPCParams,
       LQRSolution,
       MPCController
       

const StateLike = Union{MechanismState, LCPSim.StateRecord}

struct LQRSolution{T} <: Function
    Q::Matrix{T}
    R::Matrix{T}
    K::Matrix{T}
    S::Matrix{T}
    x0::Vector{T}
    u0::Vector{T}
    Δt::T
end

function LQRSolution(x0::MechanismState{T}, Q, R, Δt, contacts::AbstractVector{<:Point3D}=Point3D[]) where T
    u0 = nominal_input(x0, contacts)
    v0 = copy(velocity(x0))
    velocity(x0) .= 0
    RigidBodyDynamics.setdirty!(x0)
    K, S = LCPSim.ContactLQR.contact_dlqr(x0, u0, Q, R, Δt, contacts)
    set_velocity!(x0, v0)
    LQRSolution{T}(Q, R, K, S, copy(Vector(x0)), copy(u0), Δt)
end

(c::LQRSolution)(x) = -c.K * (Vector(x) .- c.x0) .+ c.u0

@with_kw mutable struct MPCParams{S1 <: AbstractMathProgSolver, S2 <: AbstractMathProgSolver}
    Δt::Float64 = 0.05
    horizon::Int = 15
    mip_solver::S1
    lcp_solver::S2
end


include("Models/Models.jl")
using .Models

include("mpc.jl")
include("learning.jl")

function playback(vis::MechanismVisualizer, results::AbstractVector{<:LCPUpdate}, Δt = 0.01)
    ts = cumsum([Δt for r in results])
    animate(vis, ts, [configuration(result.state) for result in results])
end


end
