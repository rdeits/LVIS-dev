abstract type MPCSink <: Function end

struct MPCSampleSink{T} <: MPCSink
    samples::Vector{Sample{T}}
    keep_nulls::Bool

    MPCSampleSink{T}(keep_nulls=false) where {T} = new{T}([], keep_nulls)
end

Base.empty!(s::MPCSampleSink) = empty!(s.samples)

function (s::MPCSampleSink)(x::StateLike, results::MPCResults)
    if s.keep_nulls || !isnull(results.lcp_updates)
        push!(s.samples, LearningMPC.Sample(x, results))
    end
end

mutable struct PlaybackSink{T} <: MPCSink
    vis::MechanismVisualizer
    Δt::T
    last_trajectory::Vector{LCPSim.LCPUpdate{T, T, T}}

    PlaybackSink(vis::MechanismVisualizer, Δt::T) where {T} = new{T}(vis, Δt, [])
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

function live_viewer(vis::MechanismVisualizer)
    x -> begin
        set_configuration!(vis, configuration(x))
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

struct Dataset{T}
    lqrsol::LQRSolution{T}
    training_data::Vector{Sample{T}}
    validation_data::Vector{Sample{T}}
    testing_data::Vector{Sample{T}}

    Dataset(lqrsol::LQRSolution{T}) where {T} = new{T}(lqrsol, [], [], [])
end

function interval_net(widths, activation=Flux.elu)
    net = Chain([Dense(widths[i-1], widths[i], activation) for i in 2:length(widths)]...)
    loss = (x, lb, ub) -> begin
        y = net(x)
        sum(ifelse.(y .< lb, lb .- y, ifelse.(y .> ub, y .- ub, 0 .* y)))
    end
    net, loss
end

struct LearnedCost{T, F1, F2} <: Function
    lqr::LQRSolution{T}
    net::F1
    tangent_net::F2
end

LearnedCost(lqr::LQRSolution, net) = LearnedCost(lqr, net, FluxExtensions.TangentPropagator(net))

function (c::LearnedCost)(x0::StateLike, results::AbstractVector{<:LCPSim.LCPUpdate})
    lqr = c.lqr
    lqrcost = sum((r.state.state .- lqr.x0)' * lqr.Q * (r.state.state .- lqr.x0) +
                  (r.input .- lqr.u0)' * lqr.R * (r.input .- lqr.u0)
                  for r in results)
    q0, q = c.tangent_net(Vector(x0))
    lqrcost + Flux.Tracker.data(q0)[] + vec(Flux.Tracker.data(q))' * Vector(results[end].state)
end
