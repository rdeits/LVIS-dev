
using MLDataPattern
using ProgressMeter
using JLD
import PyCall
using DrakeVisualizer, GeometryTypes, CoordinateTransformations
DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window()

include("../../pympc.jl")
colmat = PyMPC.colmat

sys = PyMPC.Models.BoxAtlas()

vis = Visualizer()[:boxatlas]
setgeometry!(vis, sys)

controller = PyMPC.Models.controller(sys, Q=10*eye(10))

struct Sample
    utraj::Vector{Vector{Float64}}
    xtraj::Vector{Vector{Float64}}
    switching_sequence::Vector{Int}
    u_sensitivity::Matrix{Float64}
end

num_samples = 100
num_sim_steps = 20
data = Vector{Sample}()
# data = Vector{Tuple{Vector{Float64}, Matrix{Float64}}}()

@showprogress for i in 1:num_samples
    while true
        x0 = PyMPC.Models.generate_x0(sys)
        settransform!(vis, sys, x0)
        utraj, xtraj, switching_sequence, J = PyMPC.run_mpc(controller, x0)
        u = utraj[1]
        if !isnan(u[1])
            @assert x0 == xtraj[1]
            PyMPC.Models.playback(vis, sys, xtraj, 1)
            push!(data, Sample(utraj, xtraj, collect(switching_sequence), J[1:length(u), :]))
            for j in 1:num_sim_steps
                x = xtraj[2]
                utraj = vcat(utraj[2:end], [zeros(utraj[end])])
                xnext = try
                    vec(sys.sys[:pwa_system][:simulate](colmat(xtraj[end]), [colmat(utraj[end-1])])[1][2])
                catch e
                    if isa(e, PyCall.PyError)
                        zeros(xtraj[end])
                    else
                        rethrow(e)
                    end
                end
                xtraj = vcat(xtraj[2:end], [xnext])
                switching_sequence = (switching_sequence[2:end]..., switching_sequence[end])
                utraj, xtraj, switching_sequence, J = PyMPC.run_mpc(controller, x, utraj, xtraj, switching_sequence)
                u = utraj[1]
                if isnan(u[1])
                    break
                end
                @assert x == xtraj[1]
                push!(data, Sample(utraj, xtraj, collect(switching_sequence), J[1:length(u), :]))
                PyMPC.Models.playback(vis, sys, xtraj, 1)
            end
            break
        end
    end
    save("box_atlas_100_traj.jld", "data", data)
end

save("box_atlas_100_traj.jld", "data", data)


