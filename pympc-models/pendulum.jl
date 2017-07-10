@with_kw type PendulumParams{T}
    m::T = 1.0
    l::T = 1.0
    g::T = 9.81
    t_s::T = 0.05
    bin_angle::T = π/2
    x_min::Vector{T} = [-2π, -10π]
    x_max::Vector{T} = .-x_min
    u_min::Vector{T} = [-1.0]
    u_max::Vector{T} = .-u_min
end

struct Pendulum{T} <: AbstractMPCModel
    sys::PyObject
    params::PendulumParams{T}
    X::Vector{PyObject}
    U::Vector{PyObject}
end

Δt(sys::Pendulum) = sys.params.t_s

function dynamics(p::PendulumParams, x, u)
    @unpack m, g, l = p
    q = x[1] - π
    v = x[2]
    q̇ = v
    q̈ = (u[1] - m * g * l * sin(q)) / (m * l^2)
    [q̇, q̈]
end

function Pendulum(params::PendulumParams=PendulumParams{Float64}())
    @unpack m, l, g, t_s, bin_angle, x_min, x_max, u_min, u_max = params

    # discretization method
    method = "explicit_euler"

    angle_centers = x_min[1]:bin_angle:x_max[1]

    systems = PyObject[]
    state_domains = PyObject[]
    input_domains = PyObject[]


    for θ in angle_centers
        u = [0.0]
        x = [θ, 0.0]
        A = ForwardDiff.jacobian(x -> dynamics(params, x, u), x)
        B = ForwardDiff.jacobian(u -> dynamics(params, x, u), u)
        c = dynamics(params, x, u) - (A * x + B * u)
        @assert isapprox(A * x + B * u + c, dynamics(params, x, u), atol=1e-12)
        push!(systems, 
            PyMPC.dynamical_systems.DTAffineSystem[:from_continuous](
                A, B, colmat(c), t_s, method))
        push!(state_domains,
            PyMPC.geometry.Polytope[:from_bounds](
                [θ - bin_angle/2 x_min[2]]',
                [θ + bin_angle/2 x_max[2]]')[:assemble]())
        push!(input_domains,
            PyMPC.geometry.Polytope[:from_bounds](
                colmat(u_min), colmat(u_max))[:assemble]())
    end

    pwa_sys = PyMPC.dynamical_systems.DTPWASystem[:from_orthogonal_domains](
        systems, state_domains, input_domains)

    Pendulum(pwa_sys, params, state_domains, input_domains)
end

function controller(sys::Pendulum; 
    N=10, Q=10 * eye(2), R = 0.1 * eye(1))
    objective_norm = "two"

    @unpack x_min, x_max, u_min, u_max, bin_angle, t_s = sys.params

    x = [0.0, 0]
    u = [0.0]
    A = ForwardDiff.jacobian(x -> dynamics(sys.params, x, u), x)
    B = ForwardDiff.jacobian(u -> dynamics(sys.params, x, u), u)
    c = dynamics(sys.params, x, u) - A * x - B * u
    @assert A * x + B * u + c ≈ dynamics(sys.params, x, u)
    method = "explicit_euler"
    Sf = PyMPC.dynamical_systems.DTAffineSystem[:from_continuous](
        A, B, colmat(c), t_s, method)
    # Xf = PyMPC.geometry.Polytope[:from_bounds](
    #     [0 - bin_angle/2 x_min[2]]',
    #     [0 + bin_angle/2 x_max[2]]')[:assemble]()
    Xf = PyMPC.geometry.Polytope[:from_bounds](
        colmat(x_min), colmat(x_max))[:assemble]()
    Uf = PyMPC.geometry.Polytope[:from_bounds](
        colmat(u_min), colmat(u_max))[:assemble]()
    # terminal set and cost
    P, K = PyMPC.dynamical_systems.dare(Sf[:A], Sf[:B], Q, R)
    # X_N = PyMPC.dynamical_systems.moas_closed_loop(Sf[:A], Sf[:B], K, Xf, Uf)

    # hybrid controller
    controller = PyMPC.control.MPCHybridController(sys.sys, N, objective_norm, Q, R, P, Xf)
end

function update(sys::Pendulum, x, u)
    x_next = vec(sys.sys[:simulate](colmat(x), [colmat(u)])[1][2])
end

function simulate(sys::Pendulum, x0, controller::PyObject; kwargs...)
    simulate(sys, x0, x -> vec(controller[:feedback](colmat(x))); kwargs...)
end

function simulate(sys::Pendulum, x0, controller::Function; N_sim::Int=100)
    u = typeof(x0)[]
    x = typeof(x0)[]
    push!(x, x0)
    for k in 1:N_sim
        push!(u, controller(x[end]))
        x_next = update(sys, x[end], u[end])
        push!(x, x_next)
    end
    x
end

function generate_x0(sys::Pendulum)
    x_min = [-π, -π]
    x_max = .-x_min
    rand(2) .* (x_max .- x_min) .+ x_min
end

function setgeometry!(vis::Visualizer, sys::Pendulum)
    delete!(vis)
    setgeometry!(vis[:base], HyperRectangle(Vec(-0.1, -0.1, -0.1), Vec(0.2, 0.2, 0.2)))
    setgeometry!(vis[:pendulum], HyperRectangle(Vec(0., -0.02, -0.02), Vec(sys.params.l, 0.04, 0.04)))
end

function settransform!(vis::Visualizer, sys::Pendulum, x::AbstractVector)
    settransform!(vis[:pendulum], LinearMap(RotY(-x[1] - pi/2)))
end    
