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
import ConditionalJuMP

export playback

const StateLike = Union{MechanismState, LCPSim.StateRecord}

include("mpc.jl")
include("learning.jl")

function playback(vis::MechanismVisualizer, results::AbstractVector{<:LCPUpdate}, Δt = 0.01)
    ts = cumsum([Δt for r in results])
    animate(vis, ts, [configuration(result.state) for result in results])
end


end
