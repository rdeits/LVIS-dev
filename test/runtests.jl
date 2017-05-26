using Base.Test
using Nets
import ReverseDiff

@testset "feedforward" begin
    shapes = [(1, 1), (1, 1)]
    net = Nets.Net(shapes)
    srand(1)
    for i in 1:100
        params = randn(net)
        x = randn(1)
        @test @inferred(Nets.predict(net, params, x))[1] == @inferred(Nets.predict_sensitivity(net, params, x)[1])

        dx = [1e-3]
        y = Nets.predict_sensitivity(net, params, x)
        J = y[:, 2:end]
        @test Nets.predict(net, params, x .+ dx) ≈ y[:,1] .+ J * dx
    end
end

@testset "random shapes" begin
    srand(10)
    for i in 1:100
        widths = [rand(1:5) for i in 1:4]
        net = Nets.Net(widths)
        x = randn(widths[1])
        params = randn(net)
        @test Nets.predict(net, params, x) == Nets.predict_sensitivity(net, params, x)[1:widths[end]]
        for i in 1:length(x)
            dx = zeros(x)
            dx[i] = 1e-3
            out = Nets.predict_sensitivity(net, params, x)
            y = out[:, 1]
            J = out[:, 2:end]
            @test Nets.predict(net, params, x .+ dx) ≈ y .+ J * dx
        end
    end
end

@testset "compiled jacobian" begin
    srand(20)
    for i in 1:10
        widths = [rand(1:5) for i in 1:4]
        net = Nets.Net(widths)
        g = (params, x) -> sum(Nets.predict(net, params, x))
        params = randn(net)
        x = randn(widths[1])
        g_tape = ReverseDiff.compile(ReverseDiff.GradientTape(g, (params, x)))
        results = (similar(params), similar(x))

        for i in 1:50
            x = randn(widths[1])
            params = randn(net)
            ReverseDiff.gradient!(results, g_tape, (params, x))
            @test all(results .≈ ReverseDiff.gradient(g, (params, x)))
        end
    end

    for i in 1:10
        widths = [rand(1:5) for i in 1:4]
        net = Nets.Net(widths)
        g = (params, x) -> sum(Nets.predict_sensitivity(net, params, x))
        params = randn(net)
        x = randn(widths[1])
        g_tape = ReverseDiff.compile(ReverseDiff.GradientTape(g, (params, x)))
        results = (similar(params), similar(x))

        for i in 1:50
            x = randn(widths[1])
            params = randn(net)
            ReverseDiff.gradient!(results, g_tape, (params, x))
            @test all(results .≈ ReverseDiff.gradient(g, (params, x)))
        end
    end
end
