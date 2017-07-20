using CoordinateTransformations
using DrakeVisualizer
using Gurobi

DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window()

include("boxatlas.jl")

model = Box.BoxAtlas(stiffness=1.0, damping=1.0, viscous_friction=100.)

vis = Visualizer()[:boxatlas]
setgeometry!(vis, model)

state = Box.State(vcat(rand(2), randn(8), 5 * randn(11)))
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
    
state = Box.State(
    [0.7, 0.75, 0.2, -0.75, -0.2, -0.75, 0.4, 0.1, -0.4, 0.1, zeros(11)...]
)
# state.position[Box.Trunk] = [0.5, state.position[Box.Trunk][2]]
state.velocity[Box.Trunk] = [-2, 0]
model.stiffness=1000
model.viscous_friction = 1
model.Δt = 0.05
us, xs = Box.run_mpc(model, state, 10, solver=GurobiSolver(TimeLimit=300))

for x in xs
    settransform!(vis, model, x)
    sleep(0.1)
end

us_next, xs_next = Box.run_mpc(model, xs[end], 10, solver=GurobiSolver(TimeLimit=300))
us = vcat(us, us_next)
xs = vcat(xs, xs_next)
for x in xs
    settransform!(vis, model, x)
    sleep(0.1)
end
