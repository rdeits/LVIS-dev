module MPC

using JuMP

immutable CTLinearSytstem{T}
    A::Matrix{T}
    B::Matrix{T}
end

immutable DTLinearSystem{T}
    A::Matrix{T}
    B::Matrix{T}
    Δt::T
end

num_states(s::DTLinearSystem) = size(s.A, 2)
num_inputs(s::DTLinearSystem) = size(s.B, 2)

function update(sys::DTLinearSystem, t, state, u)
    sys.A * state + sys.B * u
end

function output end


"""
Convert continuous-time linear system to discrete time, assuming
a zero-order hold on the inputs.

Based on Python code originally written by Tobia Marcucci.
"""
function discretize{T}(s::CTLinearSytstem{T}, Δt)
    nx = size(s.A, 1)
    nu = size(s.B, 2)
    c = zeros(nx + nu, nx + nu)
    c[1:nx, 1:nx] .= s.A
    c[1:nx, nx+1:end] .= s.B
    d = expm(c * Δt)
    A = d[1:nx, 1:nx]
    B = d[1:nx, nx+1:end]
    DTLinearSystem{T}(A, B, Δt)
end

function simulate(sys, controller, state, times)
    states = []
    push!(states, state)
    for t in times[1:(end-1)]
        u = output(controller, t, states[end])
        push!(states, update(sys, t, states[end], u))
    end
    states
end

immutable MPCModel
    model::Model
    times::Vector{Float64}
    x::Array{Variable, 2}
    u::Array{Variable, 2}
end



function MPCModel(sys::DTLinearSystem, N::Integer;
                  Q=eye(num_states(sys)), 
                  R=eye(num_inputs(sys)), 
                x_goal=zeros(num_states(sys)))
    model = Model()
    @variable model u[1:num_inputs(sys), 1:N]
    @variable model x[1:num_states(sys), 1:N+1]
    times = 0:sys.Δt:N * sys.Δt
    for j = 1:N
        @constraint model x[:, j+1] .== update(sys, times[j], x[:, j], u[:, j])
    end
    @objective model Min sum(first(x[:, i]' * Q * x[:, i]) for i in 2:N+1) + sum(first(u[:, i]' * R * u[:, i]) for i in 1:N)
    MPCModel(model, times, x, u)
end 

end

