using CoordinateTransformations
using DrakeVisualizer
using Gurobi

DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window()

include("boxatlas.jl")

model = Box.BoxAtlas(stiffness=1.0, damping=1.0, viscous_friction=100.)

vis = Visualizer()[:boxatlas]
setgeometry!(vis, model)

state = Box.State(vcat(rand(2), randn(8), 5 * randn(10)))
input = Box.Input(zeros(8))
model.stiffness = 10.
model.damping = 10
model.gravity = 10.0
model.Δt = 0.002
for i in 1:1000
    settransform!(vis, model, state)
    model.stiffness += 10
    model.damping = 1.0 * sqrt(model.stiffness)
    state, up, accel = Box.update(model, state, input)
    sleep(0.001)
end
    

state.position[Box.Trunk] = [0.5, state.position[Box.Trunk][2]]
state.velocity[Box.Trunk] = [-1.8, 0]
model.stiffness=1000
model.Δt = 0.05
us, xs = Box.run_mpc(model, state, 10, solver=GurobiSolver(TimeLimit=60))

for x in xs
    settransform!(vis, model, x)
    sleep(0.1)
end

# state = xs[end]
# us, xs = Box.run_mpc(model, state, 6, solver=GurobiSolver(TimeLimit=60))

# for x in xs
#     settransform!(vis, model, x)
#     sleep(0.1)
# end
