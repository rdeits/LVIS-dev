struct HybridCartPole{T} <: AbstractMPCModel
    sys::PyObject
    l::T
    d::T
    Δt::T
    X::Vector{PyObject}
    U::Vector{PyObject}
end

Δt(sys::HybridCartPole) = sys.Δt

function HybridCartPole(;
    mc = 1.,
    mp = 1.,
    l = 1.,
    d = 1.,
    k = 100.,
    g = 10.,
    t_s = .05)

    # discretization method
    method = "explicit_euler"

    # dynamics n.1
    A_1 = [
        0. 0. 1. 0.
        0. 0. 0. 1.
        0. (g*mp)/mc 0. 0.
        k/(l*mp) (g*mp^2 + g*mc*mp - k*l*mc)/(l*mc*mp) 0. 0.
        ]
    B_1 = colmat([
        0.
        0.
        1./mc
        1./(l*mc)
        ])
    c_1 = colmat([
        0.
        0.
        0.
        (d*k)/(l*mp)
        ])
    S_1 = PyMPC.dynamical_systems.DTAffineSystem[:from_continuous](A_1, B_1, c_1, t_s, method)

    # dynamics n.2
    A_2 = [
        0. 0. 1. 0.
        0. 0. 0. 1.
        0. (g*mp)/mc 0. 0.
        0. g*(mc+mp)/(l*mc) 0. 0.
        ]
    B_2 = colmat([
        0.
        0.
        1./mc
        1./(l*mc)
        ])
    c_2 = zeros(4,1)
    S_2 = PyMPC.dynamical_systems.DTAffineSystem[:from_continuous](A_2, B_2, c_2, t_s, method)

    # dynamics n.3
    A_3 = [
        0. 0. 1. 0.
        0. 0. 0. 1.
        0. (g*mp)/mc 0. 0.
        k/(l*mp) (g*mp^2 + g*mc*mp - k*l*mc)/(l*mc*mp) 0. 0.
        ]
    B_3 = colmat([
        0.
        0.
        1./mc
        1./(l*mc)
        ])
    c_3 = colmat([
        0.
        0.
        0.
        -(d*k)/(l*mp)
        ])


    S_3 = PyMPC.dynamical_systems.DTAffineSystem[:from_continuous](A_3, B_3, c_3, t_s, method)

    # list of dynamics
    S = [S_1, S_2, S_3]

    # state bounds
    x_max = [2*d, pi/8, 2.5, 2*pi]
    x_min = -x_max

    # state domain n.1
    lhs_1 = [1. -l 0. 0.]
    rhs_1 = [-d]
    X_1 = PyMPC.geometry.Polytope[:from_bounds](colmat(x_min), colmat(x_max))
    X_1[:add_facets](lhs_1, colmat(rhs_1))
    X_1[:assemble]()

    # state domain n.2
    lhs_2 = [-1. l 0. 0.
             1. -l 0. 0.]
    rhs_2 = [d, d]
    X_2 = PyMPC.geometry.Polytope[:from_bounds](colmat(x_min), colmat(x_max))
    X_2[:add_facets](lhs_2, colmat(rhs_2))
    X_2[:assemble]()

    # state domain n.3
    lhs_3 = [-1. l 0. 0.]
    rhs_3 = [-d]
    X_3 = PyMPC.geometry.Polytope[:from_bounds](colmat(x_min), colmat(x_max))
    X_3[:add_facets](lhs_3, colmat(rhs_3))
    X_3[:assemble]()

    # list of state domains
    X = [X_1, X_2, X_3]

    # input domain
    u_max = [100.]
    u_min = -u_max
    U = PyMPC.geometry.Polytope[:from_bounds](colmat(u_min), colmat(u_max))
    U[:assemble]()
    U = [U, U, U]

    pwa_sys = PyMPC.dynamical_systems.DTPWASystem[:from_orthogonal_domains](S, X, U)
    HybridCartPole(pwa_sys,
        l, 
        d,
        t_s,
        X,
        U)
end

function controller(sys::HybridCartPole; 
    N=10, Q=10 * eye(4), R = 0.1 * eye(1))
    objective_norm = "two"
    S = sys.sys[:affine_systems]
    X = sys.X
    U = sys.U
    # terminal set and cost
    P, K = PyMPC.dynamical_systems.dare(S[2][:A], S[2][:B], Q, R)
    X_N = PyMPC.dynamical_systems.moas_closed_loop(S[2][:A], S[2][:B], K, X[2], U[2])

    # hybrid controller
    controller = PyMPC.control.MPCHybridController(sys.sys, N, objective_norm, Q, R, P, X_N)
end

function update(sys::HybridCartPole, x, u)
    x_next = vec(sys.sys[:simulate](colmat(x), [colmat(u)])[1][2])
end

function simulate(sys::HybridCartPole, x0, controller::PyObject; kwargs...)
    simulate(sys, x0, x -> vec(controller[:feedback](colmat(x))); kwargs...)
end

function simulate(sys::HybridCartPole, x0, controller::Function; N_sim::Int=100)
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

function generate_x0(sys::HybridCartPole)
    x_max = [2*sys.d, pi/8, 2.5, 2*pi]
    x_min = -x_max
    rand(4) .* (x_max - x_min) .+ x_min
end

function setgeometry!(vis::Visualizer, sys::HybridCartPole)
    delete!(vis)
    setgeometry!(vis[:cart], HyperRectangle(Vec(-0.1, -0.1, -0.1), Vec(0.2, 0.2, 0.2)))
    setgeometry!(vis[:cart][:pole], HyperRectangle(Vec(0., -0.02, -0.02), Vec(sys.l, 0.04, 0.04)))
    for side in [:r, :l]
        setgeometry!(vis[:walls][side], HyperRectangle(Vec(-0.01, -sys.l/2, -sys.l/2), Vec(0.02, sys.l, sys.l)))
    end
    settransform!(vis[:walls][:r], Translation(sys.d, 0, sys.l))
    settransform!(vis[:walls][:l], Translation(-sys.d, 0, sys.l))
end

function settransform!(vis::Visualizer, sys::HybridCartPole, x::AbstractVector)
    settransform!(vis[:cart], Translation(x[1], 0, 0))
    settransform!(vis[:cart][:pole], LinearMap(RotY(-x[2] - pi/2)))
end    
