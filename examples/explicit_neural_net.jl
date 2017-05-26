using Plots
pyplot()

using Polyhedra, CDDLib
using Colors: distinguishable_colors
using ForwardDiff

import Nets

weights = [eye(2), [1. 2]]
biases = [[0., 0], [0.]]
params = Nets.Params(weights, biases)

bounds = hrep(polyhedron(SimpleVRepresentation([-1. -1; -1 1; 1 1; 1 -1]), CDDLibrary()))
results = Nets.explore(params, bounds, [1., 1])

xx = linspace(-1, 1)
yy = linspace(-1, 1)
plt = surface(xx, yy, (x, y) -> Nets.predict(params, [x, y])[1],
    legend=nothing)
for (i, (state, p)) in enumerate(results)
    V = SimpleVRepresentation(vrep(polyhedron(p, CDDLibrary()))).V
    x = V[[1:end; 1], 1]
    y = V[[1:end; 1], 2]
    plot!(plt, x, y, [Nets.predict(params, [x[i], y[i]])[1]+0.05 for i in eachindex(x)], linecolor=:white, linewidth=5)
end
display(plt)
gui()
savefig(joinpath(@__DIR__, "explicit_neural_net.svg"))
savefig(joinpath(@__DIR__, "explicit_neural_net.png"))
savefig(joinpath(@__DIR__, "explicit_neural_net.pdf"))
