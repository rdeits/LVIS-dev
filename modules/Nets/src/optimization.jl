@with_kw type SGDOpts
    learning_rate::Float64 = 0.01
    momentum::Float64 = 0.0
    batch_size::Int = 1
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
            # @assert isapprox(gradient_result[1], ReverseDiff.gradient(w -> loss(w, x, y), params), atol=1e-6)
            dw .+= sample_weight .* gradient_result[1]
        end
        for i in eachindex(params)
            v = learning_rate * dw[i] + opts.momentum * last_descent[i]
            params[i] -= v
            last_descent[i] = v
        end
    end
    params
end

@with_kw type AdamOpts
    learning_rate::Float64 = 0.01
    batch_size::Int = 1
end

struct AdamOptimizer{T, Tape <: ReverseDiff.CompiledTape}
    opts::AdamOpts
    updater::StochasticOptimization.Adam{T}
    ∇::Vector{T}
    loss_tape::Tape
    gradient_result::Tuple{Vector{T}, Vector{T}, Matrix{T}}
end

function AdamOptimizer(loss::Function, opts::AdamOpts, net::Net{<:Params{T}}, x0, y0) where T
    updater = StochasticOptimization.Adam(T)
    StochasticOptimization.init(updater, net.params.data)
    ∇ = zeros(net.params.data)
    loss_tape = ReverseDiff.compile(ReverseDiff.GradientTape(loss,
        (net.params.data, x0, y0)))
    gradient_result = (similar(net.params.data), zeros(x0), zeros(y0))
    AdamOptimizer{T, typeof(loss_tape)}(opts, updater, ∇, loss_tape, gradient_result)
end

function update!(params::AbstractVector, adam::AdamOptimizer, data::AbstractVector{<:Tuple})
    sample_weight = 1 / adam.opts.batch_size
    for batch in batchview(shuffleobs(data), adam.opts.batch_size)
        adam.∇ .= 0
        for (x, y) in batch
            ReverseDiff.gradient!(adam.gradient_result, adam.loss_tape, (params, x, y))
            adam.∇ .+= sample_weight .* adam.gradient_result[1]
        end
        StochasticOptimization.update!(params, adam.updater, adam.∇, adam.opts.learning_rate)
    end
    params
end

