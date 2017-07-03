module Models

import PyMPC
using PyMPC: colmat
using PyCall: PyObject
using DrakeVisualizer, GeometryTypes, CoordinateTransformations
using Parameters
using ForwardDiff
import DrakeVisualizer: setgeometry!, settransform!

include("hybrid_cart_pole.jl")
include("pendulum.jl")

end
