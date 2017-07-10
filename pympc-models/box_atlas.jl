@pyimport pympc.box_atlas_pwa_dynamics as boxatlas

immutable BoxAtlas <: AbstractMPCModel
    sys::PyObject
    Δt::Float64

    BoxAtlas() = new(boxatlas.pwa_system, boxatlas.t_s)
end

Δt(sys::BoxAtlas) = sys.Δt

function controller(sys::BoxAtlas;
                    N = 10,
                    Q = eye(10),
                    R = eye(9),
                    objective_norm = "two")
    P = Q
    X_N = PyMPC.polytope.Polytope[:from_bounds](-5 .* ones(10, 1), 5 .* ones(10, 1))[:assemble]()
    PyMPC.control.MPCHybridController(sys.sys, N, objective_norm, Q, R, P, X_N)
end

function setgeometry!(vis::Visualizer, sys::BoxAtlas)
    delete!(vis)
    setgeometry!(vis[:body], HyperRectangle(Vec(-0.1, -0.05, -0.2), Vec(0.2, 0.1, 0.4)))
    setgeometry!(vis[:lf], HyperSphere(Point(0., 0, 0), 0.05))
    setgeometry!(vis[:rf], HyperSphere(Point(0., 0, 0), 0.05))
    setgeometry!(vis[:hand], HyperSphere(Point(0., 0, 0), 0.05))

    v = vis[:environment]
    setgeometry!(v[:floor], HyperRectangle(Vec(-1., -1, -0.001), Vec(2., 2., 0.002)))
    settransform!(v[:floor], Translation(1.0, 0, 0))
    setgeometry!(v[:wall], HyperRectangle(Vec(-0.001, -1, -1), Vec(0.002, 2, 2)))
    settransform!(v[:wall], Translation(0.0, 0, 1.0))
end

function settransform!(vis::Visualizer, sys::BoxAtlas, y::AbstractArray, 
                        # u::AbstractArray
                        )
    x = vec(y) + vec(boxatlas.x_eq)
    settransform!(vis[:body], Translation(x[1], 0, x[2]))
    settransform!(vis[:lf], Translation(x[3], 0, x[4]))
    settransform!(vis[:rf], Translation(x[5], 0, x[6]))
    settransform!(vis[:hand], Translation(x[7], 0, x[8]))

    # if x[4] < 0
    #     setgeometry!(vis[:lf][:force], PolyLine([[0., 0, 0], 
    #         0.1 * [u[7], 0, boxatlas.stiffness * (0 - x[4])]], radius=0.02, end_head=ArrowHead()))
    # else
    #     delete!(vis[:lf][:force])
    # end
    # if x[6] < 0
    #     setgeometry!(vis[:rf][:force], PolyLine([[0., 0, 0], 
    #         0.1 * [u[8], 0, boxatlas.stiffness * (0 - x[6])]], radius=0.02, end_head=ArrowHead()))
    # else
    #     delete!(vis[:rf][:force])
    # end
    # if x[7] < 0
    #     setgeometry!(vis[:hand][:force], PolyLine([[0., 0, 0], 
    #         0.1 * [boxatlas.stiffness * (0 - x[7]), 0, u[9]]], radius=0.02, end_head=ArrowHead()))
    # else
    #     delete!(vis[:hand][:force])
    # end
end 

function generate_x0(sys::BoxAtlas)
    i = 0
    while true
        i += 1
        x0 = zeros(10)
        x0[1:2] .= 2 .* (rand(2) .- 0.5)
        x0[3:4] .= x0[1:2] .- rand(2)
        x0[5:6] .= x0[1:2] .+ rand(2)
        x0[7] = x0[1] - rand()
        x0[8] = x0[2] + (rand() - 0.5)
        x0[9:10] = 2 .* (rand(2) .- 0.5)
        # x_max = ones(10)
        # x_min = .-x_max
        # x0 = rand(10) .* (x_max .- x_min) .+ x_min
        try
            xtraj, ss = sys.sys[:simulate](colmat(x0), [zeros(9, 1)])
            return x0, i
        catch e
            if !isa(e, PyCall.PyError)
                rethrow(e)
            end
        end
    end
end

