module Box

using JuMP
using Polyhedra
using DrakeVisualizer
using CoordinateTransformations: Translation
using Parameters: @with_kw
using CDDLib: CDDLibrary
using StaticArrays: SVector
using Gurobi: GurobiSolver
using ConditionalJuMP: @disjunction, setup_indicators!
import Base: convert, vec

function from_bounds(lb::AbstractVector, ub::AbstractVector)
    len = length(lb)
    @assert length(lb) == length(ub)
    SimpleHRepresentation(vcat(eye(len), .-eye(len)), vcat(ub, .-lb))
end

@enum Body Trunk LeftFoot RightFoot LeftHand RightHand

@with_kw mutable struct BoxAtlas{T}
    position_limits::Dict{Body, SimpleHRepresentation{2, T}} = Dict(
        Trunk=>from_bounds([-1.0, 0], [1.0, 2.0]),
        LeftFoot=>from_bounds([0.0, -1.0], [0.4, -0.5]),
        RightFoot=>from_bounds([-0.4, -1.0], [0.0, -0.5]),
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

function Base.similar(x::State{T}, ::Type{T2}=T) where {T, T2}
    position = Dict{Body, SVector{2, T2}}()
    for (k, v) in x.position
        position[k] = zeros(SVector{2, T2})
    end
    velocity = Dict{Body, SVector{2, T2}}()
    for (k, v) in x.velocity
        velocity[k] = zeros(SVector{2, T2})
    end
    State(position, velocity)
end

JuMP.getvalue(s::State) = State(
    Dict([(body, getvalue.(p)) for (body, p) in s.position]),
    Dict([(body, getvalue.(p)) for (body, p) in s.velocity]))

vec(s::State) = convert(Array, vcat(
    s.position[Trunk],
    s.position[LeftFoot],
    s.position[RightFoot],
    s.position[LeftHand],
    s.position[RightHand],
    s.velocity[Trunk],
    s.velocity[LeftFoot],
    s.velocity[RightFoot],
    s.velocity[LeftHand],
    s.velocity[RightHand]))

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

JuMP.getvalue(s::Input) = Input(
    Dict([(body, getvalue.(p)) for (body, p) in s.force]))

vec(u::Input) = convert(Array, vcat(
    u.force[LeftFoot],
    u.force[RightFoot],
    u.force[LeftHand],
    u.force[RightHand]))

struct HRepIter{P <: HRepresentation}
    p::P
end

Base.start(h::HRepIter) = starthrep(h.p)
Base.done(h::HRepIter, i) = donehrep(h.p, i)
Base.next(h::HRepIter, i) = nexthrep(h.p, i)
Base.length(h::HRepIter) = length(h.p)

struct StateUpdate{T}
    input_forces::Dict{Body, SVector{2, T}}
    joint_limit_forces::Dict{Body, SVector{2, T}}
    gravity_forces::Dict{Body, SVector{2, T}}
    ground_contact_forces::Dict{Body, SVector{2, T}}
    wall_contact_forces::Dict{Body, SVector{2, T}}
    damping_forces::Dict{Body, SVector{2, T}}

    function StateUpdate{T}() where {T}
        bodies = [Trunk, LeftFoot, RightFoot, LeftHand, RightHand]
        up = new{T}(
           Dict([body=>zeros(SVector{2, T}) for body in bodies]),
           Dict([body=>zeros(SVector{2, T}) for body in bodies]),
           Dict([body=>zeros(SVector{2, T}) for body in bodies]),
           Dict([body=>zeros(SVector{2, T}) for body in bodies]),
           Dict([body=>zeros(SVector{2, T}) for body in bodies]),
           Dict([body=>zeros(SVector{2, T}) for body in bodies]),
        )
        for body in bodies
            up.input_forces[body] = zeros(SVector{2, T})
            up.joint_limit_forces[body] = zeros(SVector{2, T})
            up.gravity_forces[body] = zeros(SVector{2, T})
            up.ground_contact_forces[body] = zeros(SVector{2, T})
            up.wall_contact_forces[body] = zeros(SVector{2, T})
            up.damping_forces[body] = zeros(SVector{2, T})
        end
        up
    end
end

function accelerations(robot::BoxAtlas, up::StateUpdate{T}) where {T}
    bodies = [Trunk, LeftFoot, RightFoot, LeftHand, RightHand]
    accelerations = Dict([body=>zeros(SVector{2, T}) for body in bodies])
    for body in bodies
        accelerations[body] = sum(getfield(up, field)[body] for field in fieldnames(up)) ./ robot.masses[body]
    end
    for body in [LeftFoot, RightFoot, LeftHand, RightHand]
        # Non-inertial frame
        accelerations[body] -= accelerations[Trunk]
    end
    accelerations
end

function update(model::BoxAtlas, x::State{T1}, u::Input{T2}) where {T1, T2}
    Tnext = Base.promote_op(+, T1, T2)
    xnext = similar(x, Tnext)

    up = StateUpdate{Tnext}()
    for (body, force) in u.force
        up.input_forces[body] = force
        up.input_forces[Trunk] -= force
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
        up.joint_limit_forces[body] = force
        up.joint_limit_forces[Trunk] -= force
    end

    # Gravity
    for body in [Trunk, keys(u.force)...]
        up.gravity_forces[body] = SVector(0, -model.gravity)
    end

    # Ground contact
    for body in (LeftFoot, RightFoot)
        separation = x.position[Trunk][2] + x.position[body][2]
        up.ground_contact_forces[body] = @disjunction if separation <= 0
            (-separation .* model.stiffness .* SVector(0., 1) - model.viscous_friction .* (x.velocity[Trunk] .+ x.velocity[body]) .* SVector(1., 1))
        else
            zeros(SVector{2, Tnext})
        end
    end

    # Wall contact
    body = RightHand
    separation = x.position[Trunk][1] + x.position[body][1]
    up.wall_contact_forces[body] = @disjunction if separation <= 0
        (-separation .* model.stiffness .* SVector(1., 0) - model.viscous_friction .* (x.velocity[Trunk] .+ x.velocity[body]) .* SVector(1., 1))
    else
        zeros(SVector{2, Tnext})
    end

    # Damping
    for body in [Trunk, keys(u.force)...]
        damping_force = .-model.damping .* x.velocity[body]
        up.damping_forces[body] = damping_force
        up.damping_forces[Trunk] -= damping_force
    end

    # # Apply soft velocity limits
    # for body in keys(u.force)
    #     force = zero(SVector{2, Tnext})
    #     for face in HRepIter(model.velocity_limits[body])
    #         separation = -(face.a' * x.velocity[body] - face.β)
    #         force += @disjunction if separation <= 0
    #             separation .* model.stiffness .* face.a
    #         else
    #             zeros(face.a)
    #         end
    #     end
    #     acceleration[body] += force ./ model.masses[body]
    #     acceleration[Trunk] += -force ./ model.masses[Trunk]
    # end

    # # Non-inertial reference frame
    # for body in keys(u.force)
    #     acceleration[body] -= acceleration[Trunk]
    # end
    acceleration = accelerations(model, up)

    for body in [Trunk, keys(u.force)...]
        xnext.velocity[body] = x.velocity[body] .+ acceleration[body] .* model.Δt
        xnext.position[body] = x.position[body] .+ x.velocity[body] .* model.Δt .+ 0.5 .* acceleration[body] .* model.Δt.^2
    end
    xnext, up, acceleration
end

function DrakeVisualizer.setgeometry!(vis::Visualizer, model::BoxAtlas)
    delete!(vis)
    setgeometry!(vis[:trunk], HyperRectangle(Vec(-0.1, -0.1, -0.1), Vec(0.2, 0.2, 0.2)))
    for body in (LeftFoot, RightFoot, LeftHand, RightHand)
        setgeometry!(vis[:trunk][Symbol(body)], HyperSphere(Point(0., 0, 0), 0.05))
    end
    setgeometry!(vis[:environment][:right_wall], HyperRectangle(Vec(-0.01, -0.5, 0), Vec(0.02, 1.0, 1.5)))
end

function DrakeVisualizer.settransform!(vis::Visualizer, model::BoxAtlas, x::State)
    settransform!(vis[:trunk], Translation(x.position[Trunk][1], 0, x.position[Trunk][2]))
    for body in (LeftFoot, RightFoot, LeftHand, RightHand)
        settransform!(vis[:trunk][Symbol(body)], Translation(x.position[body][1], 0, x.position[body][2]))
    end
end

function setlimits!(u::Input{Variable}, robot::BoxAtlas)
    for (body, force) in u.force
        setlowerbound.(force, -robot.effort_limits[body])
        setupperbound.(force, robot.effort_limits[body])
    end
end

function setlimits!(x::State{Variable}, robot::BoxAtlas)
    for (body, position) in x.position
        verts = SimpleVRepresentation(polyhedron(robot.position_limits[body], CDDLibrary())).V
        lb = vec(minimum(verts, 1))
        ub = vec(maximum(verts, 1))
        widths = ub .- lb
        lb .-= widths
        ub .+= widths
        setlowerbound.(position, lb)
        setupperbound.(position, ub)
        @assert all(getlowerbound.(position) .== lb)
        @assert all(getupperbound.(position) .== ub)
    end

    for (body, velocity) in x.velocity
        verts = SimpleVRepresentation(polyhedron(robot.velocity_limits[body], CDDLibrary())).V
        lb = vec(minimum(verts, 1))
        ub = vec(maximum(verts, 1))
        widths = ub .- lb
        lb .-= widths
        ub .+= widths
        setlowerbound.(velocity, lb)
        setupperbound.(velocity, ub)
        @assert all(getlowerbound.(velocity) .== lb)
        @assert all(getupperbound.(velocity) .== ub)
    end
end

function run_mpc(robot::BoxAtlas, x0::State, N=10; 
        solver=GurobiSolver(), 
        xdesired=State([0.5, 0.9, 0.2, -0.9, -0.2, -0.9, 0.4, 0.1, -0.4, 0.1, zeros(10)...]))
    model = Model(solver=solver)
    u = Input(@variable(model, [1:10], basename="u"))
    setlimits!(u, robot)
    x = State(@variable(model, [1:20], basename="x"))
    setlimits!(x, robot)
    xnext, up, accel = update(robot, x0, u)
    @constraint(model, vec(x) .== vec(xnext))
    inputs = [u]
    states = [x]
    updates = [up]
    accels = [accel]
    for i in 2:N
        u = Input(@variable(model, [1:10], basename="u"))
        setlimits!(u, robot)
        x = State(@variable(model, [1:20], basename="x"))
        setlimits!(x, robot)
        xnext, up, accel = update(robot, states[end], u)
        @constraint(model, vec(x) .== vec(xnext))
        push!(inputs, u)
        push!(states, x)
        push!(updates, up)
        push!(accels, accel)
    end
    setup_indicators!(model)
    obj = 0.1 * sum([sum(vec(u).^2) for u in inputs]) + 10 * sum((vec(states[end]) .- vec(xdesired)).^2)
    for up in updates
        for body in keys(u.force)
            obj += up.ground_contact_forces[body][1]^2
            obj += up.wall_contact_forces[body][2]^2
        end
    end
    @objective(model, Min, obj)
    status = solve(model)
    getvalue.(inputs), [x0, getvalue.(states)...]
end


end
    