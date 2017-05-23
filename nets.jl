module Nets

import ReverseDiff
using MLDataPattern: batchview, shuffleobs

head(t::Tuple) = tuple(t[1])

function viewblocks{T <: NTuple}(data::AbstractArray, shapes::AbstractVector{T})
    starts = cumsum(vcat([1], prod.(shapes)))
    [reshape(view(data, starts[i]:(starts[i+1] - 1)), shapes[i]) for i in 1:length(shapes)]
end

type PANet{Sensitive}
    shapes::Vector{NTuple{2, Int}}
end

nweights(net::PANet) = sum(prod, net.shapes)
nbiases(net::PANet) = sum(first, net.shapes)
nparams(net::PANet) = nweights(net) + nbiases(net)
Base.rand(net::PANet) = rand(nparams(net))
Base.randn(net::PANet) = randn(nparams(net))

function predict(net::PANet{true}, params::AbstractVector, x::AbstractVector)
    weights = viewblocks(params, net.shapes)
    biases = viewblocks(@view(params[(nweights(net) + 1):end]), head.(net.shapes))
    y = similar(x, Base.promote_eltype(params, x), (length(x), 1))
    y .= x
    J = eye(eltype(y), length(x))
    for i in 1:(length(net.shapes) - 1)
        w = weights[i]
        y = w * y .+ biases[i]
        J = w * J
        z = [yy >= 0 ? 1.0 : 0.1 for yy in y]
        y = y .* z
        J = J .* z
    end
    w = weights[end]
    vcat(vec(w * y), vec(w * J))
end

function predict(net::PANet{false}, params::AbstractVector, x::AbstractVector)
    weights = viewblocks(params, net.shapes)
    biases = viewblocks(@view(params[(nweights(net) + 1):end]), head.(net.shapes))
    y = similar(x, Base.promote_eltype(params, x), (length(x), 1))
    y .= x
    for i in 1:(length(net.shapes) - 1)
        w = weights[i]
        y = w * y .+ biases[i]
        z = [yy >= 0 ? 1.0 : 0.1 for yy in y]
        y = y .* z
    end
    w = weights[end]
    vec(w * y)
end

function sgd!(loss, params, data, lr=0.01, momentum=0.8, batchsize=1)
    last_descent = zeros(params)
    dw = zeros(params)
    dw_sample = zeros(dw)
    sample_weight = 1 / batchsize
    for batch in batchview(shuffleobs(data), batchsize)
        dw .= 0
        for (x, y) in batch
            ReverseDiff.gradient!(dw_sample, w -> loss(w, x, y), params)
            if any(isnan.(dw_sample))
                @show params x y dw_sample
                error("nan")
            end
            dw .+= sample_weight .* dw_sample
        end
        for i in eachindex(params)
            v = lr * dw[i] + momentum * last_descent[i]
            params[i] -= v
            last_descent[i] = v
        end
    end
    params
end

end