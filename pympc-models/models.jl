module Models

import PyMPC
using PyMPC: colmat
using PyCall
using DrakeVisualizer, GeometryTypes, CoordinateTransformations
using Parameters
using ForwardDiff
import DrakeVisualizer: setgeometry!, settransform!

abstract type AbstractMPCModel end

function playback(vis::Visualizer, sys::AbstractMPCModel, xs::AbstractVector, realtime_rate=1.0)
    for x in xs
        settransform!(vis, sys, x)
        sleep(Δt(sys) / realtime_rate)
    end
end

include("hybrid_cart_pole.jl")
include("pendulum.jl")
include("box_atlas.jl")

end
