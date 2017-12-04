__precompile__()

module BoxValkyries

using RigidBodyDynamics
using DrakeVisualizer
using RigidBodyTreeInspector
using LCPSim
using Polyhedra
using CDDLib
using StaticArrays: SVector
using Gurobi
using JuMP
using LearningMPC
using ExplicitQPs
using Parameters
using ConditionalJuMP

struct BoxValkyrie{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
end

const urdf = joinpath(@__DIR__, "box_valkyrie.urdf")

function DrakeVisualizer.setgeometry!(basevis::Visualizer, boxval::BoxValkyrie)
    vis = basevis[:robot]
    setgeometry!(vis, boxval.mechanism, parse_urdf(urdf, boxval.mechanism))

    wall_radius = 1.5
    bounds = SimpleHRepresentation(vcat(eye(3), -eye(3)), vcat([wall_radius + 0.1, 0.5, 2.0], -[-wall_radius - 0.1, -0.5, -0.1]))
    for (body, contacts) in boxval.environment.contacts
        for obstacle in contacts.obstacles
            addgeometry!(basevis[:environment], CDDPolyhedron{3, Float64}(intersect(obstacle.interior, bounds)))
        end
    end
end

function BoxValkyrie()
    urdf_mech = parse_urdf(Float64, urdf)
    mechanism, base = planar_revolute_base()
    attach!(mechanism, base, urdf_mech)
    world = root_body(mechanism)


    # http://nvlpubs.nist.gov/nistpubs/jres/28/jresv28n4p439_A1b.pdf
    floor = planar_obstacle(default_frame(world), [0, 0, 1.], [0, 0, 0.], 2.0)
    wall = planar_obstacle(default_frame(world), [1., 0, 0], [-1.0, 0, 0], 0.5)

    # contact_limbs = findbody.(mechanism, ["rh", "lh", "rf", "lf"])
    # hands = findbody.(mechanism, ["rh", "lh"])
    # feet = findbody.(mechanism, ["rf", "lf"])
    rf = findbody(mechanism, "rf")
    lf = findbody(mechanism, "lf")
    rh = findbody(mechanism, "rh")
    lh = findbody(mechanism, "lh")

    env = Environment(
        Dict(
             rf => ContactEnvironment(
                [Point3D(default_frame(rf), SVector(0., 0, 0))],
                [floor]),
             lf => ContactEnvironment(
                [Point3D(default_frame(lf), SVector(0., 0, 0))],
                [floor, wall]),
             lh => ContactEnvironment(
                [Point3D(default_frame(lh), SVector(0., 0, 0))],
                [wall]),
             ))

    BoxValkyrie(mechanism, env)
end

function nominal_state(boxval::BoxValkyrie)
    mechanism = boxval.mechanism
    xstar = MechanismState{Float64}(mechanism)
    set_configuration!(xstar, findjoint(mechanism, "base_z"), [1.05])
    set_configuration!(xstar, findjoint(mechanism, "core_to_lf_extension"), [0.8])
    set_configuration!(xstar, findjoint(mechanism, "core_to_rf_extension"), [0.8])
    set_configuration!(xstar, findjoint(mechanism, "core_to_lh_extension"), [0.5])
    set_configuration!(xstar, findjoint(mechanism, "core_to_rh_extension"), [0.5])
    xstar
end

function default_costs(boxval::BoxValkyrie)
    x = nominal_state(boxval)

    qq = zeros(num_positions(x))
    qq[configuration_range(x, findjoint(x.mechanism, "base_x"))]        .= 0
    qq[configuration_range(x, findjoint(x.mechanism, "base_z"))]        .= 10
    qq[configuration_range(x, findjoint(x.mechanism, "base_rotation"))] .= 500
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rh_extension"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lh_extension"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rh_rotation"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lh_rotation"))]  .= 0.5
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rf_extension"))]  .= 0.1
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lf_extension"))]  .= 0.1
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_rf_rotation"))]  .= 0.1
    qq[configuration_range(x, findjoint(x.mechanism, "core_to_lf_rotation"))]  .= 0.1

    qv = fill(1e-3, num_velocities(x))
    qv[velocity_range(x, findjoint(x.mechanism, "base_x"))] .= 10
    qv[velocity_range(x, findjoint(x.mechanism, "base_z"))] .= 1
    qv[velocity_range(x, findjoint(x.mechanism, "base_rotation"))] .= 20

    Q = diagm(vcat(qq, qv))
    
    # minimize (rx - lx)^2 = rx^2 - 2rxlx + lx^2
    rx = configuration_range(x, findjoint(x.mechanism, "core_to_rf_extension"))
    lx = configuration_range(x, findjoint(x.mechanism, "core_to_lf_extension"))
    w_centering = 1
    Q[rx, rx] += w_centering
    Q[lx, lx] += w_centering
    Q[lx, rx] -= w_centering
    Q[rx, lx] -= w_centering
    rθ = configuration_range(x, findjoint(x.mechanism, "core_to_rf_rotation"))
    lθ = configuration_range(x, findjoint(x.mechanism, "core_to_lf_rotation"))
    w_centering = 1
    Q[rθ, rθ] += w_centering
    Q[lθ, lθ] += w_centering
    Q[lθ, rθ] -= w_centering
    Q[rθ, lθ] -= w_centering

    rr = fill(0.002, num_velocities(x))
    R = diagm(rr)
    Q, R
end

end
