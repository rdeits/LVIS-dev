module Explicit

using Polyhedra
using CDDLib
using DiffBase
using ForwardDiff
using Nets: Params, ninputs

leaky_relu(y, active::Bool=(y >= 0)) = active ? y : 0.1 * y

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

end