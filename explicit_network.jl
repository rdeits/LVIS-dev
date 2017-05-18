module ExNet

using Polyhedra
using ForwardDiff
using CDDLib

immutable Net{T}
    coefficients::Vector{Matrix{T}}
    biases::Vector{Vector{T}}

    function (::Type{Net{T}}){T}(c::Vector, b::Vector)
        @assert(length(c) == length(b))
        new{T}(c, b)
    end
end

Net{T1, T2}(c::Vector{Matrix{T1}}, b::Vector{Vector{T2}}) = Net{promote_type(T1, T2)}(c, b)

num_inputs(n::Net) = size(n.coefficients[1], 1)
num_outputs(n::Net) = size(n.coefficients[end], 2)

function remove_small_weights!(net::Net, threshold=1e-16)
    for weights in [net.coefficients, net.biases]
        for layer in weights
            for i in 1:length(layer)
                if abs(layer[i]) < threshold
                    layer[i] = 0
                end
            end
        end
    end
end

leaky_relu(slope) = (input, active=input >= 0) -> active ? input : slope * input

function feedforward{T}(n::Net{T}, x::Vector, relu_activations=nothing)
    R = promote_type(T, eltype(x))
    result = Vector{R}()
    current_layer = convert(Vector{R}, x)
    j_relu = 1
    for i in 1:length(n.coefficients) - 1
        current_layer = n.coefficients[i]' * current_layer
        current_layer .+= n.biases[i]
        append!(result, current_layer)
        if typeof(relu_activations) === Void
            current_layer .= leaky_relu(0.1).(current_layer)
        else
            current_layer .= leaky_relu(0.1).(current_layer, relu_activations[j_relu:(j_relu + length(current_layer) - 1)])
            j_relu += length(current_layer)
        end
    end
    append!(result, n.coefficients[end]' * current_layer .+ n.biases[end])
    result
end

function relu_constraints{T}(net::Net{T}, relu_activations)
    x = zeros(T, num_inputs(net))
    y = feedforward(net, x, relu_activations)
    out = DiffBase.DiffResult(similar(y), similar(y, length(y), length(x)))
    ForwardDiff.jacobian!(out, y -> feedforward(net, y, relu_activations), x)
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
    A = J[1:end-1, :]
    b = -v[1:end-1]
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

function explore{T}(net::Net{T}, bounds, start::Vector)
    value = feedforward(net, start)
    state = value[1:(end - num_outputs(net))] .>= 0
    constr = relu_constraints(net, state)
    results = Dict{typeof(state), typeof(constr)}()

    active_set = Set([state])

    while !isempty(active_set)
        new_active_set = Set{typeof(state)}()
        for state in active_set
            constr = relu_constraints(net, state)
            p = intersect(constr, bounds)
            if isempty(SimpleVRepresentation(vrep(polyhedron(p, CDDLibrary(:exact)))).V)
                continue
            end
            # if isempty(polyhedron(p, CDDLibrary()), CDDSolver())
            #     continue
            # end
            results[state] = p
            for i in 1:(length(p) - length(bounds))
                # if !sredundant(p, i)[1]
                    newstate = copy(state)
                    newstate[i] = !newstate[i]
                    if !haskey(results, newstate)
                        push!(new_active_set, newstate)
                    end
                # end
            end
        end
        active_set = new_active_set
    end
    results
end

end
