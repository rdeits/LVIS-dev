module Learning

import MXNet.mx: AbstractDataProvider,
                 AbstractDataBatch,
                 get_batch_size,
                 provide_data,
                 provide_label,
                 get_data,
                 get_label,
                 count_samples
using MLDataPattern
using StaticArrays: SVector

immutable Sample{Nx, Nu}
    x0::SVector{Nx, Float64}
    u0::SVector{Nu, Float64}
end

immutable SampleProvider{V <: MLDataPattern.BatchView} <: AbstractDataProvider
    obsview::V
    nx::Int
    nu::Int
end

immutable SampleBatch{T <: Tuple} <: AbstractDataBatch
    data::T
end

get_batch_size(p::SampleProvider) = MLDataPattern.batchsize(p.obsview)
provide_data(p::SampleProvider) = [(:x, (p.nx, batchsize(p.obsview)))]
provide_label(p::SampleProvider) = [(:u_label, (p.nu, batchsize(p.obsview)))]

Base.eltype(p::SampleProvider) = eltype(p.obsview)
Base.start(p::SampleProvider) = start(p.obsview)
Base.done(p::SampleProvider, state) = done(p.obsview, state)
function Base.next(p::SampleProvider, state)
    item, newstate = next(p.obsview, state)
    SampleBatch(item), newstate
end

count_samples(p::SampleProvider, b::SampleBatch) = size(b.data[1], 2)
get_data(p::SampleProvider, b::SampleBatch) = b.data[1]
get_label(p::SampleProvider, b::SampleBatch) = b.data[2]


# immutable SampleBatch{Nx, Nu}
#     samples::SubArray{Sample{Nx, Nu}

# immutable SampleProvider{Nx, Nu}
#     samples::Vector{Sample{Nx, Nu}}
#     minibatch_size::Int
# end

# MXNet.get_batch_size(p::SampleProvider) = p.minibatch_size
# MXNet.provide_data{Nx, Nu}(p::SampleProvider{Nx, Nu}) = [(:x, (Nx,))]
# MXNet.provide_label{Nx, Nu}(p::SampleProvider{Nx, Nu}) = [(:u, (Nu,))]

# Base.eltype(SampleProvider



end