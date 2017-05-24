module Nets

using Parameters
import ReverseDiff
using MLDataPattern: batchview, shuffleobs

head(t::Tuple) = tuple(t[1])

function viewblocks{T <: NTuple}(data::AbstractArray, shapes::AbstractVector{T})
    starts = cumsum(vcat([1], prod.(shapes)))
    [reshape(view(data, starts[i]:(starts[i+1] - 1)), shapes[i]) for i in 1:length(shapes)]
end

type Net
    shapes::Vector{NTuple{2, Int}}
end

Net(widths::Vector{<:Integer}) = Net(collect(zip(widths[2:end], widths[1:end-1])))

nweights(net::Net) = sum(prod, net.shapes)
nbiases(net::Net) = sum(first, net.shapes)
nparams(net::Net) = nweights(net) + nbiases(net)
ninputs(net::Net) = net.shapes[1][2]
noutput(net::Net) = net.shapes[end][1]
Base.rand(net::Net) = rand(nparams(net))
Base.randn(net::Net) = randn(nparams(net))

@ReverseDiff.forward relu(y) = y >= 0 ? y : 0.1 * y
@ReverseDiff.forward relu_sensitivity(y, j) = y >= 0 ? j : 0.1 * j

function predict_sensitivity(net::Net, params::AbstractVector, x::AbstractVector)
    weights = viewblocks(params, net.shapes)
    biases = viewblocks(@view(params[(nweights(net) + 1):end]), head.(net.shapes))
    J = similar(weights[1])
    y = weights[1] * x .+ biases[1]
    J .= weights[1]
    for i in 2:length(net.shapes)
        y = relu.(y)
        J = relu_sensitivity.(y, J)

        y = weights[i] * y + biases[i]
        J = weights[i] * J
    end
    vcat(vec(y), vec(J))
end

function predict(net::Net, params::AbstractVector, x::AbstractVector)
    weights = viewblocks(params, net.shapes)
    biases = viewblocks(@view(params[(nweights(net) + 1):end]), head.(net.shapes))
    y = weights[1] * x .+ biases[1]
    for i in 2:length(net.shapes)
        y = relu.(y)

        y = weights[i] * y + biases[i]
    end
    vec(y)
end

@with_kw type SGDOpts
    learning_rate::Float64 = 0.01
    momentum::Float64 = 0.0
    batch_size::Int = 1
    learning_decay = 1.0
end

function sgd!(loss, params, data, opts::SGDOpts=SGDOpts())
    last_descent = zeros(params)
    dw = zeros(params)
    sample_weight = 1 / opts.batch_size
    n_in = length(data[1][1])
    n_out = length(data[1][2])
    loss_tape = ReverseDiff.compile(ReverseDiff.GradientTape(loss, 
        (params, randn(n_in), randn(n_out))))
    gradient_result = (similar(params), zeros(n_in), zeros(n_out))
    learning_rate = opts.learning_rate
    for batch in batchview(shuffleobs(data), opts.batch_size)
        dw .= 0
        for (x, y) in batch
            ReverseDiff.gradient!(gradient_result, loss_tape, (params, x, y))
            @assert gradient_result[1] ≈ ReverseDiff.gradient(w -> loss(w, x, y), params)
            dw .+= sample_weight .* gradient_result[1]
        end
        for i in eachindex(params)
            v = learning_rate * dw[i] + opts.momentum * last_descent[i]
            params[i] -= v
            last_descent[i] = v
        end
        learning_rate *= opts.learning_decay
    end
    params
end

using Base.Test

@testset "feedforward" begin
    shapes = [(1, 1), (1, 1)]
    net = Net(shapes)
    srand(1)
    for i in 1:100
        params = randn(net)
        x = randn(1)
        @test @inferred(predict(net, params, x))[1] == @inferred(predict_sensitivity(net, params, x)[1])

        dx = [1e-3]
        y = predict_sensitivity(net, params, x)
        J = y[2:2]
        @test predict(net, params, x .+ dx) ≈ y[1:1] .+ J .* dx
    end
end

@testset "random shapes" begin
    srand(10)
    for i in 1:100
        widths = [rand(1:5) for i in 1:4]
        net = Net(widths)
        x = randn(widths[1])
        params = randn(net)
        @test predict(net, params, x) == predict_sensitivity(net, params, x)[1:widths[end]]
        for i in 1:length(x)
            dx = zeros(x)
            dx[i] = 1e-3
            out = predict_sensitivity(net, params, x)
            y = out[1:widths[end]]
            J = reshape(out[widths[end]+1:end], widths[end], length(x))
            @test predict(net, params, x .+ dx) ≈ y .+ J * dx
        end
    end
end

@testset "compiled jacobian" begin
    srand(20)
    for i in 1:10
        widths = [rand(1:5) for i in 1:4]
        net = Net(widths)
        g = (params, x) -> sum(predict(net, params, x))
        params = randn(net)
        x = randn(widths[1])
        g_tape = ReverseDiff.compile(ReverseDiff.GradientTape(g, (params, x)))
        results = (similar(params), similar(x))

        for i in 1:50
            x = randn(widths[1])
            params = randn(net)
            ReverseDiff.gradient!(results, g_tape, (params, x))
            @test all(results .≈ ReverseDiff.gradient(g, (params, x)))
        end
    end

    for i in 1:10
        widths = [rand(1:5) for i in 1:4]
        net = Net(widths)
        g = (params, x) -> sum(predict_sensitivity(net, params, x))
        params = randn(net)
        x = randn(widths[1])
        g_tape = ReverseDiff.compile(ReverseDiff.GradientTape(g, (params, x)))
        results = (similar(params), similar(x))

        for i in 1:50
            x = randn(widths[1])
            params = randn(net)
            ReverseDiff.gradient!(results, g_tape, (params, x))
            @test all(results .≈ ReverseDiff.gradient(g, (params, x)))
        end
    end
end



end