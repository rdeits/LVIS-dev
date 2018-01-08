module Nets2

using ForwardDiff
using ReverseDiff
using ReverseDiff: @forward

abstract type Layer <: Function end

params(f) = ()
with_params(f) = f

struct Affine{TA <: AbstractMatrix, Tb <: AbstractVector} <: Layer
    A::TA
    b::Tb
end

params(a::Affine) = (a.A, a.b)
with_params(a::Affine, A, b) = Affine(A, b)

(a::Affine)(x) = a.A * x .+ a.b

function (a::Affine)(x)
    [@view(a.A[i, :])' * x + a.b[i] for i in 1:size(a.A, 1)]
end

struct Chain{L <: Tuple}
    layers::L
end

Chain(layers...) = Chain{typeof(layers)}(layers)

params(c::Chain) = _params(c.layers...)
_params(layer, layers...) = (params(layer)..., _params(layers...)...)
_params() = ()

(c::Chain)(x) = _chain_call(x, c.layers...)

_chain_call(x, layer, layers...) = _chain_call(layer(x), layers...)
_chain_call(x) = x

with_params(c::Chain, params...) = Chain(_with_params(c.layers, params)...)

function _with_params(layers, parameters)
    layer = first(layers)
    layer_params, remaining_params = _split(params(layer), parameters)
    (with_params(layer, layer_params...), _with_params(Base.tail(layers), remaining_params)...)
end

_with_params(::Tuple{}, ::Tuple{}) = ()


function _split(t1::NTuple{N1, Any}, t2::NTuple{N2, Any}) where {N1, N2}
    beginning, remainder = _split(Base.tail(t1), Base.tail(t2))
    (first(t2), beginning...), remainder
end

_split(::Tuple{}, t2::NTuple{N2, Any}) where {N2} = ((), t2)

relu(y) = y >= 0 ? y : zero(y)
leaky_relu(y) = y >= 0 ? y : 0.1 * y
elu(y) = y >= 0 ? y : exp(y) - 1
gaussian(y) = exp(-y^2)

activation(func) = x -> @forward(func).(x)

struct TangentPropagation{F}
	inner::F
end

(p::TangentPropagation)(x) = (p.inner(x), ForwardDiff.jacobian(p.inner, x))

params(p::TangentPropagation) = params(p.inner)
with_params(p::TangentPropagation, parameters...) = TangentPropagation(with_params(p.inner, parameters...))

function initialize!(f, params::Tuple)
	for p in params
		initialize!(f, p)
	end
end

function initialize!(f, a::AbstractArray)
	for I in eachindex(a)
		a[I] = f()
	end
end

end