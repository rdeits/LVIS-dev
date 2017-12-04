__precompile__()

module Nets

export Net, Params, predict, predict_sensitivity

using Parameters: @with_kw
using ReverseDiff
using ReverseDiff: @forward
import StochasticOptimization
using CoordinateTransformations: AffineMap, UniformScaling, transform_deriv
using MLDataPattern: batchview, shuffleobs

head(t::Tuple) = tuple(t[1])

function viewblocks(data::AbstractArray, shapes::AbstractVector{T}) where {T <: NTuple}
    starts = cumsum(vcat([1], prod.(shapes)))
    [reshape(view(data, starts[i]:(starts[i+1] - 1)), shapes[i]) for i in 1:length(shapes)]
end

struct Params{T, D <: AbstractVector{T}, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    data::D
    weights::Vector{M}
    biases::Vector{V}
    shapes::Vector{NTuple{2, Int}}

    Params{T, D, M, V}(data::D, weights::Vector{M}, biases::Vector{V}) where {T, D, M, V} =
        new{T, D, M, V}(data, weights, biases, size.(weights))
end

Params(
    data::D, weights::Vector{M}, biases::Vector{V}) where {T,D <: AbstractVector{T},M <: AbstractMatrix{T},V <: AbstractVector{T}} =
    Params{T, D, M, V}(data, weights, biases)

function Params(shapes::Vector{<:NTuple}, data::AbstractVector)
    weights = viewblocks(data, shapes)
    biases = viewblocks(@view(data[(nweights(shapes) + 1):end]), head.(shapes))
    Params(data, weights, biases)
end

Params(widths::AbstractVector{<:Integer}, data::AbstractVector) =
    Params(collect(zip(widths[2:end], widths[1:end-1])), data)

ninputs(p::Params) = size(p.weights[1], 2)
noutputs(p::Params) = size(p.weights[end], 1)
shapes(p::Params) = p.shapes
nparams(widths::AbstractVector{<:Integer}) = nparams(collect(zip(widths[2:end], widths[1:end-1])))
nweights(shapes::AbstractVector{<:NTuple{2, Integer}}) = sum(prod, shapes)
nbiases(shapes::AbstractVector{<:NTuple{2, Integer}}) = sum(first, shapes)
nparams(shapes::AbstractVector{<:NTuple{2, Integer}}) = nweights(shapes) + nbiases(shapes)

Base.zeros(::Type{Params{T}}, widths::AbstractVector{<:Integer}) where {T} =
    Params(widths, zeros(T, nparams(widths)))
Base.rand(::Type{Params{T}}, widths::AbstractVector{<:Integer}) where {T} =
    Params(widths, rand(T, nparams(widths)))
Base.randn(::Type{Params{T}}, widths::AbstractVector{<:Integer}) where {T} =
    Params(widths, randn(T, nparams(widths)))

struct Net{P <: Params, F, T <: AffineMap} <: Function
    params::P
    activation::F
    input_tform::T
    output_tform::T
end

Net(params::Params, activation=leaky_relu) =
    Net(params,
        activation,
        AffineMap(UniformScaling(1.0), zeros(ninputs(params))),
        AffineMap(UniformScaling(1.0), zeros(noutputs(params))))

nweights(net::Net) = nweights(net.params)
nbiases(net::Net) = nbiases(net.params)
nparams(net::Net) = nparams(net.params)
ninputs(net::Net) = ninputs(net.params)
noutput(net::Net) = noutputs(net.params)
params(net::Net) = net.params
(net::Net)(x) = predict(net, x)

Base.rand(net::Net) = rand(nparams(net))
Base.randn(net::Net) = randn(nparams(net))

Base.similar(net::Net, data::AbstractVector) =
    Net(Params(shapes(params(net)), data),
        net.activation,
        net.input_tform,
        net.output_tform)

const relu_x = 1
function hat_relu(y)
    if y >= relu_x
        -(0.1/relu_x) * (y - relu_x)
    elseif y >= 0
        -(1/relu_x) * y + 1
    elseif y >= -relu_x
        (1/relu_x) * y + 1
    else
        (0.1/relu_x) * (y + relu_x)
    end
end

derivative(::typeof(hat_relu)) = y -> begin
    if y >= relu_x
        -(0.1/relu_x)
    elseif y >= 0
        -(1/relu_x)
    elseif y >= -relu_x
        (1/relu_x)
    else
        (0.1/relu_x)
    end
end

leaky_relu(y) = y >= 0 ? y : 0.1 * y
derivative(::typeof(leaky_relu)) = y -> y >= 0 ? 1.0 : 0.1

elu(y) = y >= 0 ? y : exp(y) - 1
derivative(::typeof(elu)) = y -> y >= 0 ? 1.0 : exp(y)

gaussian(y) = exp(-y^2)
derivative(::typeof(gaussian)) = y -> -2 * y * exp(-y^2)

function predict(net::Net, x::AbstractVector)
    params = net.params
    y = params.weights[1] * net.input_tform(x) + params.biases[1]
    for i in 2:length(params.weights)
        y = (@forward(net.activation)).(y)
        y = params.weights[i] * y + params.biases[i]
    end
    net.output_tform(y)
end

function predict_sensitivity(net::Net, x::AbstractVector)
    params = net.params
    y = params.weights[1] * net.input_tform(x) + params.biases[1]
    J = params.weights[1] * transform_deriv(net.input_tform, x)
    for i in 2:length(params.weights)
        dJ = (@forward(derivative(net.activation))).(y)
        J = dJ .* J
        y = (@forward(net.activation)).(y)
        y = params.weights[i] * y + params.biases[i]
        J = params.weights[i] * J
    end
    hcat(net.output_tform(y), transform_deriv(net.output_tform, y) * J)
end


include("explicit.jl")
include("optimization.jl")
include("scaling.jl")

end