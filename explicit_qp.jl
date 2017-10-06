module ExplicitQP

using JuMP
using ForwardDiff
using ForwardDiff: value, partials, jacobian

# See Bemporad, "A Survey on Explicit Model Predictive Control", 
# section 2.2 (page 353)

getcol(x::Variable) = x.col

struct DualNumber{T} <: Real
    value::T
    partials::Vector{T}
end

ForwardDiff.value(d::DualNumber) = d.value
ForwardDiff.partials(d::DualNumber) = d.partials
ForwardDiff.jacobian(v::Vector{<:DualNumber}) = hcat(partials.(v)...)'

struct ExplicitSolution{T}
    model::Model
    solution::Vector{DualNumber{T}}
    params::Vector{T}
    variable_map::Vector{Tuple{Bool, Int}}
end

function getsolution(ex::ExplicitSolution, v::Variable)
    @assert v.m === ex.model
    isparam, idx = ex.variable_map[getcol(v)]
    @assert !isparam "$v is a parameter, so you can access its value with getparameter()"
    ex.solution[idx]
end

function getparameter(ex::ExplicitSolution, v::Variable)
    @assert v.m === ex.model
    isparam, idx = ex.variable_map[getcol(v)]
    @assert isparam "$v is not a parameter, so you can access its value with getsolution()"
    ex.params[idx]
end


function variable_map(m::Model, params::AbstractArray{Variable})
    var_index = 1
    param_index = 1
    nvars = length(m.colCat)
    var_map = Tuple{Bool, Int}[]
    for i in 1:nvars
        var = Variable(m, i)
        if var in params
            push!(var_map, (true, param_index))
            param_index += 1
        else
            push!(var_map, (false, var_index))
            var_index += 1
        end
    end
    @assert param_index == (length(params) + 1)
    @assert param_index + var_index == (nvars + 2)
    var_map
end

function active_inequalities(m::Model, params::AbstractArray{Variable}, eps=1e-3)
    nvars = length(m.colCat)
    A_active = SparseVector{Float64, Int64}[]
    b_active = Float64[]
    
    for i in 1:nvars
        var = Variable(m, i)
        if var in params
            continue
        end
        λ = getdual(var)
        if abs(λ) > eps
            if λ > 0
                ai = sparsevec([i], [-1.0], nvars)
                bi = -m.colLower[i]
            else
                ai = sparsevec([i], [1.0], nvars)
                bi = m.colUpper[i]
            end 
            push!(A_active, ai)
            push!(b_active, bi)
        end
    end
    
    nconstr = length(m.linconstr)
    for (constraint, λ) in zip(m.linconstr, m.linconstrDuals)
        if abs(λ) > eps
            if λ > 0
                ai = -sparsevec([var.col for var in constraint.terms.vars], constraint.terms.coeffs, nvars)
                bi = -constraint.lb
            else
                ai = sparsevec([var.col for var in constraint.terms.vars], constraint.terms.coeffs, nvars)
                bi = constraint.ub
            end
            push!(A_active, ai)
            push!(b_active, bi)
        end
    end
    A = hcat(A_active...)'
    b = vcat(b_active...)
    A, b
end


function explicit_solution(m::Model, params::AbstractArray{Variable}, eps=1e-3)
    @assert isempty(m.obj.aff)
    nvars = length(m.colCat)
    Ã, W̃ = active_inequalities(m, params)
    
    param_cols = Set([v.col for v in params])
    isparam = collect(1:nvars) .∈ param_cols
    G̃ = Ã[:, .!isparam]
    S̃ = .-Ã[:, isparam]
    
    Q = sparse(getcol.(m.obj.qvars1), getcol.(m.obj.qvars2), m.obj.qcoeffs, nvars, nvars)
    Q = 0.5 .* (Q .+ Q')
    H = Q[.!isparam, .!isparam]
    Hi = full(H)^-1
    F = Q[.!isparam, isparam]
    Y = Q[isparam, isparam]
    
    v_test = rand(nvars)
    x_test = v_test[isparam]
    z_test = v_test[.!isparam]
    @assert 0.5 * v_test' * Q * v_test ≈ (0.5 * z_test' * H * z_test + x_test' * F' * z_test + 0.5 * x_test' * Y * x_test)

    x = getvalue(params)    
    @show size(G̃) size(Hi) size(W̃) size(S̃) size(F) x
    λ_active = -(G̃ * Hi * G̃') * (W̃ + (S̃ + G̃ * Hi * F) * x)
    T = Hi * G̃' * (G̃ * Hi * G̃')^-1

    jacobian = T * (S̃ + G̃ * Hi * F) - Hi * F
    value = T * W̃ + jacobian * x

    solution = [DualNumber(value[i], jacobian[i, :]) for i in 1:length(value)]
    ExplicitSolution(m, solution, x, variable_map(m, params))

    # ExplicitSolution(T * W̃, T * (S̃ + G̃ * Hi * F) - Hi * F, x)
end

end