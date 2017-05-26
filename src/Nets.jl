__precompile__()

module Nets

using Parameters: @with_kw
import ReverseDiff
using MLDataPattern: batchview, shuffleobs
using Polyhedra
using CDDLib
using DiffBase
using ForwardDiff

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

immutable Params{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    weights::Vector{M}
    biases::Vector{V}
end

Params{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}(weights::Vector{M}, biases::Vector{V}) = 
    Params{T, M, V}(weights, biases)

function Params(net::Net, params::AbstractVector)
    weights = viewblocks(params, net.shapes)
    biases = viewblocks(@view(params[(nweights(net) + 1):end]), head.(net.shapes))
    Params(weights, biases)
end

ninputs(p::Params) = size(p.weights[1], 2)
noutputs(p::Params) = size(p.weights[end], 1)

@ReverseDiff.forward leaky_relu(y) = y >= 0 ? y : 0.1 * y
@ReverseDiff.forward leaky_relu_sensitivity(y, j) = y >= 0 ? j : 0.1 * j

predict_sensitivity(net::Net, params::AbstractVector, x::AbstractVector) = 
    predict_sensitivity(Params(net, params), x)

function predict_sensitivity(params::Params, x::AbstractVector)
    J = similar(params.weights[1])
    y = params.weights[1] * x .+ params.biases[1]
    J .= params.weights[1]
    for i in 2:length(params.weights)
        y = leaky_relu.(y)
        J = leaky_relu_sensitivity.(y, J)
        y = params.weights[i] * y + params.biases[i]
        J = params.weights[i] * J
    end
    hcat(y, J)
end

predict(net::Net, params::AbstractVector, x::AbstractVector) = 
    predict(Params(net, params), x)

function predict(params::Params, x::AbstractVector)
    y = params.weights[1] * x .+ params.biases[1]
    for i in 2:length(params.weights)
        y = leaky_relu.(y)
        y = params.weights[i] * y + params.biases[i]
    end
    y
end

leaky_relu(y, active::Bool) = active ? y : 0.1 * y

function predict_and_record(params::Params, x::AbstractVector, relu_activations=nothing)
    y = params.weights[1] * x .+ params.biases[1]
    result = [copy(y)]
    for i in 2:length(params.weights)
        if typeof(relu_activations) == Void
            y = leaky_relu.(y)
        else
            y = leaky_relu.(y, relu_activations[i - 1])
        end
        y = params.weights[i] * y + params.biases[i]
        push!(result, copy(y))
    end
    result
end

function relu_constraints{T}(params::Params{T}, relu_activations)
    x = zeros(T, ninputs(params))
    f = x -> vcat(predict_and_record(params, x, relu_activations)[1:end-1]...)
    y = f(x)
    out = DiffBase.DiffResult(similar(y), similar(y, length(y), length(x)))
    ForwardDiff.jacobian!(out, f, x)
    v = DiffBase.value(out)
    J = DiffBase.jacobian(out)
    # y = v + J * x
    # we want to ensure that y remains on the current side of 0
    # for each relu. 
    # y_i >= 0 if relu i is active
    # y_i <= 0 otherwise
    # 
    # -y_i <= 0 if active
    #  y_i <= 0 if inactive
    #
    A = J
    b = -v
    i = 1
    for a in relu_activations
        for ai in a
            if ai
                A[i, :] .*= -1
                b[i] .*= -1
            end
            i += 1
        end
    end
    SimpleHRepresentation(A, b)
end

function explore{T}(params::Params{T}, bounds, start::AbstractVector)
    record = predict_and_record(params, start)
    state = [x .>= 0 for x in record[1:end-1]]
    constr = relu_constraints(params, state)

    results = Dict{typeof(state), typeof(constr)}()

    active_set = Set([state])

    while !isempty(active_set)
        new_active_set = Set{typeof(state)}()
        for state in active_set
            constr = relu_constraints(params, state)
            p = intersect(constr, bounds)
            if isempty(SimpleVRepresentation(vrep(polyhedron(p, CDDLibrary(:exact)))).V)
                continue
            end
            results[state] = p
            for J in eachindex(state)
                layerstate = state[J]
                for I in eachindex(layerstate)
                    newstate = deepcopy(state)
                    newstate[J][I] = !newstate[J][I]
                    if !haskey(results, newstate)
                        push!(new_active_set, newstate)
                    end
                end
            end
        end
        active_set = new_active_set
    end
    results
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
    sz_in = size(data[1][1])
    sz_out = size(data[1][2])
    loss_tape = ReverseDiff.compile(ReverseDiff.GradientTape(loss, 
        (params, randn(sz_in), randn(sz_out))))
    gradient_result = (similar(params), zeros(sz_in), zeros(sz_out))
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

function runtests()
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
            J = y[:, 2:end]
            @test predict(net, params, x .+ dx) ≈ y[:,1] .+ J * dx
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
                y = out[:, 1]
                J = out[:, 2:end]
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

end