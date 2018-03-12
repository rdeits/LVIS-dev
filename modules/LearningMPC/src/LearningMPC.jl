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
import Nets
import ConditionalJuMP
import ExplicitQPs

export playback

const StateLike = Union{MechanismState, LCPSim.StateRecord}

include("mpc.jl")
include("learning.jl")
include("onlinempc.jl")

end
