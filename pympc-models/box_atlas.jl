@pyimport pympc.models.boxatlas as boxatlas

immutable BoxAtlas <: AbstractMPCModel
    sys::PyObject

    BoxAtlas(;kwargs...) = new(boxatlas.BoxAtlasPWAModel(;kwargs...))
end

Δt(sys::BoxAtlas) = sys.sys[:t_s]

function controller(sys::BoxAtlas;
                    N = 10,
                    Q = eye(10),
                    R = eye(9),
                    objective_norm = "two")
    sys.sys[:controller](N=N, Q=Q, R=R, objective_norm=objective_norm)
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
    x_eq, u_eq = sys.sys[:equilibrium_point]()
    x = vec(y) + vec(x_eq)
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

function generate_x0(sys::BoxAtlas, controller::PyObject)
    vec(sys.sys[:random_state](controller=controller))
end

function generate_x0(sys::BoxAtlas)
    vec(sys.sys[:random_state]())
end

