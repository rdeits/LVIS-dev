module PiecewiseFunctions
    import Base: broadcast
    export PiecewiseFunction, from_above, from_below

    immutable PiecewiseFunction{Breaks, F} <: Function
        breaks::Breaks
        pieces::Vector{F}

        function (::Type{PiecewiseFunction{B, F}}){B, F}(breaks::B, pieces::Vector{F})
            @assert issorted(breaks)
            @assert length(breaks) == length(pieces) + 1
            new{B, F}(breaks, pieces)
        end
    end

    PiecewiseFunction{Breaks, F}(breaks::Breaks, pieces::Vector{F}) = PiecewiseFunction{Breaks, F}(breaks, pieces)

    (pf::PiecewiseFunction)(t) = from_above(pf, t)

    function from_above(pf::PiecewiseFunction, t)
        i = searchsortedlast(pf.breaks, t)
        if i <= 0 || i >= length(pf.breaks)
            error("Input value $t is out of the allowable range [$(pf.breaks[1]), $(pf.breaks[end]))")
        end
        pf.pieces[i](t - pf.breaks[i])
    end

    function from_below(pf::PiecewiseFunction, t)
        i = searchsortedfirst(pf.breaks, t)
        if i <= 1 || i >= length(pf.breaks) + 1
            error("Input value $t is out of the allowable range ($(pf.breaks[1]), $(pf.breaks[end])]")
        end
        pf.pieces[i - 1](t - pf.breaks[i - 1])
    end

    broadcast(f, pf::PiecewiseFunction) = PiecewiseFunction(pf.breaks, f.(pf.pieces))
end

pf = PiecewiseFunctions