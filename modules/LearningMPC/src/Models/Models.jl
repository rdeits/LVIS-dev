module Models

using RigidBodyDynamics
using LCPSim
using LearningMPC
using MeshCat
using MeshCatMechanisms
using Gurobi

export Hopper,
       CartPole,
       BoxAtlas,
       AbstractModel,
       mechanism,
       environment,
       nominal_state,
       default_costs,
       urdf


abstract type AbstractModel{T} end

function nominal_state end
function default_costs end
function mechanism end
function environment end
function urdf end


MeshCatMechanisms.MechanismVisualizer(h::AbstractModel, v::Visualizer=Visualizer()) = MechanismVisualizer(mechanism(h), URDFVisuals(urdf(h)), v)

include("cartpole.jl")
include("boxatlas.jl")
include("hopper.jl")

end