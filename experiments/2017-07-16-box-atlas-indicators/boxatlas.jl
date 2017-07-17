module Box

using Parameters
using Polyhedra
using StaticArrays
using JuMPIndicators
using DrakeVisualizer, CoordinateTransformations
import Base: convert

function from_bounds(lb::AbstractVector, ub::AbstractVector)
    len = length(lb)
    @assert length(lb) == length(ub)
    SimpleHRepresentation(vcat(eye(len), .-eye(len)), vcat(ub, .-lb))
end

@enum Body Trunk LeftFoot RightFoot LeftHand RightHand

@with_kw mutable struct BoxAtlas{T}
    position_limits::Dict{Body, SimpleHRepresentation{2, T}} = Dict(
        Trunk=>from_bounds([0.3, 0.3], [0.7, 0.7]),
        LeftFoot=>from_bounds([0.0, -0.7], [0.4, -0.3]),
        RightFoot=>from_bounds([-0.4, -0.7], [0.0, -0.3]),
        LeftHand=>from_bounds([0.2, -0.1], [0.6, 0.3]),
        RightHand=>from_bounds([-0.6, -0.1], [-0.2, 0.3])
    )
    velocity_limits::Dict{Body, SimpleHRepresentation{2, T}} = Dict(
        Trunk=>from_bounds([-5., -5], [5, 5]),
        LeftFoot=>from_bounds([-5., -5], [5, 5]),
        RightFoot=>from_bounds([-5., -5], [5, 5]),
        LeftHand=>from_bounds([-5., -5], [5, 5]),
        RightHand=>from_bounds([-5., -5], [5, 5]),
        )
    effort_limits::Dict{Body, T} = Dict(
        LeftFoot=>150.,
        RightFoot=>150.,
        LeftHand=>50.,
        RightHand=>50.
    )
    masses::Dict{Body, T} = Dict(
        Trunk=>10.,
        LeftFoot=>1.,
        RightFoot=>1.,
        LeftHand=>1.,
        RightHand=>1.
        )
    stiffness::T = 100.
    damping::T = 10.
    gravity::T = 10.
    viscous_friction::T = 100.
    Δt::T = 0.1
end

struct State{T}
    position::Dict{Body, SVector{2, T}}
    velocity::Dict{Body, SVector{2, T}}
end

convert(::Type{State}, x::AbstractVector{T}) where {T} = convert(State{T}, x)

function convert(::Type{State{T}}, x::AbstractVector) where {T}
    State{T}(
        Dict(
            Trunk => x[1:2], 
            LeftFoot => x[3:4],
            RightFoot => x[5:6],
            LeftHand => x[7:8],
            RightHand => x[9:10]
            ),
        Dict(
            Trunk => x[11:12], 
            LeftFoot => x[13:14],
            RightFoot => x[15:16],
            LeftHand => x[17:18],
            RightHand => x[19:20]
            )
        )
end

function Base.similar(x::State)
    position = similar(x.position)
    for (k, v) in x.position
        position[k] = similar(v)
    end
    velocity = similar(x.velocity)
    for (k, v) in x.velocity
        velocity[k] = similar(v)
    end
    State(position, velocity)
end

struct Input{T}
    force::Dict{Body, SVector{2, T}}
end

convert(::Type{Input}, x::AbstractVector{T}) where {T} = convert(Input{T}, x)

function convert(::Type{Input{T}}, x::AbstractVector) where {T}
    Input{T}(
        Dict(
             LeftFoot => x[1:2],
             RightFoot => x[3:4],
             LeftHand => x[5:6],
             RightHand => x[7:8]
            )
    )
end

struct HRepIter{P <: HRepresentation}
    p::P
end

Base.start(h::HRepIter) = starthrep(h.p)
Base.done(h::HRepIter, i) = donehrep(h.p, i)
Base.next(h::HRepIter, i) = nexthrep(h.p, i)
Base.length(h::HRepIter) = length(h.p)

function update(model::BoxAtlas, x::State{T}, u::Input{T}) where {T}
    xnext = similar(x)
    Tnext = Base.promote_op(+, T, T)
    acceleration = Dict{Body, SVector{2, Tnext}}()

    # Apply joint forces
    acceleration[Trunk] = zero(SVector{2, Tnext})
    for body in keys(u.force)
        acceleration[body] = u.force[body] ./ model.masses[body]
        acceleration[Trunk] -= u.force[body] ./ model.masses[Trunk]
    end

    # Apply soft joint limits
    for body in keys(u.force)
        force = zero(SVector{2, Tnext})
        for face in HRepIter(model.position_limits[body])
            separation = -(face.a' * x.position[body] - face.β)
            force += @disjunction if separation <= 0
                separation .* model.stiffness .* face.a
            else
                zeros(face.a)
            end
        end
        acceleration[body] += force ./ model.masses[body]
        acceleration[Trunk] += -force ./ model.masses[Trunk]
    end

    # Gravity
    for body in [Trunk, keys(u.force)...]
        acceleration[body] += SVector(0, -model.gravity)
    end

    # Ground contact
    for body in (LeftFoot, RightFoot)
        separation = x.position[Trunk][2] + x.position[body][2]
        acceleration[body] += @disjunction if separation <= 0
            (-separation .* model.stiffness .* SVector(0., 1) - model.viscous_friction .* (x.velocity[Trunk] .+ x.velocity[body]) .* SVector(1., 1)) ./ model.masses[body]
        else
            zeros(SVector{2, T})
        end
    end

    # Damping
    for body in [Trunk, keys(u.force)...]
        damping_force = .-model.damping .* x.velocity[body]
        acceleration[body] += damping_force ./ model.masses[body]
        acceleration[Trunk] -= damping_force ./ model.masses[Trunk]
    end

    # Non-inertial reference frame
    for body in keys(u.force)
        acceleration[body] -= acceleration[Trunk]
    end

    for body in [Trunk, keys(u.force)...]
        xnext.velocity[body] = x.velocity[body] .+ acceleration[body] .* model.Δt
        xnext.position[body] = x.position[body] .+ x.velocity[body] .* model.Δt .+ 0.5 .* acceleration[body] .* model.Δt.^2
    end
    xnext
end

function DrakeVisualizer.setgeometry!(vis::Visualizer, model::BoxAtlas)
    delete!(vis)
    setgeometry!(vis[:trunk], HyperRectangle(Vec(-0.1, -0.1, -0.1), Vec(0.2, 0.2, 0.2)))
    for body in (LeftFoot, RightFoot, LeftHand, RightHand)
        setgeometry!(vis[:trunk][Symbol(body)], HyperSphere(Point(0., 0, 0), 0.05))
    end
end

function DrakeVisualizer.settransform!(vis::Visualizer, model::BoxAtlas, x::State)
    settransform!(vis[:trunk], Translation(x.position[Trunk][1], 0, x.position[Trunk][2]))
    for body in (LeftFoot, RightFoot, LeftHand, RightHand)
        settransform!(vis[:trunk][Symbol(body)], Translation(x.position[body][1], 0, x.position[body][2]))
    end
end

end
    