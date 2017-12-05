__precompile__()

module CartPoles

using RigidBodyDynamics
using DrakeVisualizer
using RigidBodyTreeInspector
using LCPSim
using Polyhedra
using CDDLib

urdf = joinpath(@__DIR__, "cartpole.urdf")

struct CartPole{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
end

function DrakeVisualizer.setgeometry!(basevis::Visualizer, cartpole::CartPole)
    vis = basevis[:robot]
    setgeometry!(vis, cartpole.mechanism, parse_urdf(urdf, cartpole.mechanism))

    wall_radius = 1.5
    bounds = SimpleHRepresentation(vcat(eye(3), -eye(3)), vcat([wall_radius + 0.1, 0.5, 2.0], -[-wall_radius - 0.1, -0.5, -0.1]))
    for (body, contacts) in cartpole.environment.contacts
        for obstacle in contacts.obstacles
            addgeometry!(basevis[:environment], CDDPolyhedron{3, Float64}(intersect(obstacle.interior, bounds)))
        end
    end
end

function CartPole(walls=true)
    mechanism = parse_urdf(Float64, urdf)
    world = root_body(mechanism)

    wall_radius = 1.5
    μ = 0.5
    if walls
        walls = [planar_obstacle(default_frame(world), [1., 0, 0.], [-wall_radius, 0, 0.], μ), 
            planar_obstacle(default_frame(world), [-1., 0, 0.], [wall_radius, 0, 0.], μ)]
        pole = findbody(mechanism, "pole")
        env = Environment(
            Dict(pole => ContactEnvironment(
                    [Point3D(default_frame(pole), 0., 0., 1.)],
                    walls)))
    else
        env = Environment{Float64}(Dict())
    end
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

end
