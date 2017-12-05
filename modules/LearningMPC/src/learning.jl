function sensitive_loss(net, λ)
    nx = length(net.input_tform.v)
    nu = length(net.output_tform.v)
    q = fill(λ, 1, 1 + nx)
    q[1] = 1 - λ
    function loss(params, x, y)
        @assert size(x) == (nx,)
        @assert size(y) == (nu, 1 + nx)
        sum(abs2,
            q .* (Nets.predict_sensitivity(similar(net, params), x) .- y)
            )
    end
    return loss
end

function control_net(mechanism::Mechanism, hidden_widths::Vector, activation::Function)
    nx = num_positions(mechanism) + num_velocities(mechanism)
    nu = num_velocities(mechanism)
    widths = [nx, hidden_widths..., nu]
    x_to_u = AffineMap(eye(nx), zeros(nx))
    v_to_y = AffineMap(diagm([max(abs(b.lower), abs(b.upper)) for b in LCPSim.all_effort_bounds(mechanism)]),
                       zeros(nu))
    params = 0.1 * randn(Nets.Params{Float64}, widths).data
    Nets.Net(Nets.Params(widths, params), activation, x_to_u, v_to_y)
end

abstract type MPCSink <: Function end

struct MPCSampleSink{T} <: MPCSink
    samples::Vector{Sample{T}}

    MPCSampleSink{T}() where {T} = new{T}([])
end

Base.empty!(s::MPCSampleSink) = empty!(s.samples)

function (s::MPCSampleSink)(x::StateLike, results::MPCResults)
    if !isnull(results.lcp_updates) && !isnull(results.jacobian)
        push!(s.samples, LearningMPC.Sample(x, results))
    end
end

mutable struct PlaybackSink{T} <: MPCSink
    vis::Visualizer
    Δt::T
    last_trajectory::Vector{LCPSim.LCPUpdate{T, T, T}}

    PlaybackSink(vis::Visualizer, Δt::T) where {T} = new{T}(vis, Δt, [])
end

function (p::PlaybackSink)(x::StateLike, results::MPCResults)
    if !isnull(results.lcp_updates)
        p.last_trajectory = get(results.lcp_updates)
        playback(p.vis, p.last_trajectory, p.Δt)
    end
end

function call_each(f::Function...)
    f1 = first(f)
    f2 = Base.tail(f)
    (args...) -> begin
        result = f1(args...)
        (x -> x(args...)).(f2)
        return result
    end
end

# import Base: &
# (&)(s::MPCSink...) = (x, results) -> begin
#     for sink in s
#         sink(x, results)
#     end
# end

function live_viewer(mechanism::Mechanism, vis::Visualizer)
    state = MechanismState{Float64}(mechanism)
    x -> begin
        set_configuration!(state, configuration(x))
        settransform!(vis, state)
    end
end

function call_with_probability(args::Tuple{Function, Float64}...)
    p_total = sum(last.(args))
    p = last.(args) ./ p_total
    cdf = cumsum(p)
    @assert cdf[end] ≈ 1
    (x...) -> begin
        i = searchsortedfirst(cdf, rand())
        if i > length(cdf)
            i = length(cdf)
        end
        first(args[i])(x...)
    end
end

function dagger_controller(mpc_controller, net_controller, p_mpc)
    x ->  begin
        if rand() < p_mpc
            return mpc_controller(x)
        else
            return net_controller(x)
        end
    end
end

function randomize!(x::MechanismState, xstar::MechanismState, σ_q = 0.1, σ_v = 0.5)
    set_configuration!(x, configuration(xstar) .+ σ_q .* randn(num_positions(xstar)))
    set_velocity!(x, velocity(xstar) .+ σ_v .* randn(num_velocities(xstar)))
end

struct Snapshot{T}
    params::Vector{T}
    net::Nets.Net
end

struct Dataset{T}
    lqrsol::LQRSolution{T}
    training_data::Vector{Sample{T}}
    validation_data::Vector{Sample{T}}
    testing_data::Vector{Sample{T}}

    Dataset(lqrsol::LQRSolution{T}) where {T} = new{T}(lqrsol, [], [], [])
end

training_loss(net::Nets.Net, data::Dataset) = average_loss(net, data.training_data)
validation_loss(net::Nets.Net, data::Dataset) = average_loss(net, data.validation_data)
testing_loss(net::Nets.Net, data::Dataset) = average_loss(net, data.testing_data)

function average_loss(net::Nets.Net, samples::Vector{<:Sample})
    mean(samples) do sample
        x, uJ = features(sample)
        sum(abs2, Nets.predict(net, x) - uJ[:, 1])
    end
end


