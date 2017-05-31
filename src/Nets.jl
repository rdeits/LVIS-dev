__precompile__()

module Nets

using Parameters: @with_kw
import ReverseDiff
import StochasticOptimization
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

include("Explicit.jl")
include("optimization.jl")

end