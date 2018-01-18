__precompile__()

module FluxExtensions

using Flux
using ForwardDiff
using CoordinateTransformations

activation(σ) = x -> σ.(x)

plain(layer::Dense) = activation(layer.σ) ∘ AffineMap(Flux.Tracker.value(layer.W), Flux.Tracker.value(layer.b))
plain(t::Transformation) = t
plain(f::Function) = f
plain(chain::Chain) = reduce(∘, identity, plain.(reverse(chain.layers)))

struct TangentPropagator{F <: Function, C <: Chain}
    f::F
    chain::C
end

function TangentPropagator(chain::Chain)
    f = reduce(∘, identity, _propagate_tangent.(reverse(chain.layers)))
    TangentPropagator(x -> f((x, eye(length(x)))), chain)
end

(p::TangentPropagator)(x) = p.f(x)

Flux.params(p::TangentPropagator) = Flux.params(p.chain)

function _propagate_tangent(f)
    (xJ) -> begin
        (f(xJ[1]), ForwardDiff.jacobian(f, xJ[1]) * xJ[2])
    end
end

function _propagate_tangent(f::Dense)
    xJ -> begin
        x, J = xJ
        y = f.W * x + f.b
        gσ = ForwardDiff.derivative.(f.σ, y)
        (f(x), gσ .* f.W * J)
    end
end

end
