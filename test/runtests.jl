using Base.Test
using Nets
using CoordinateTransformations
import ReverseDiff
import ForwardDiff

@testset "feedforward" begin
widths = [1, 1]
    srand(1)
    for i in 1:100
        params = randn(Params{Float64}, widths)
        net = Net(params)
        x = randn(1)
        @test @inferred(net(x)) == @inferred(predict_sensitivity(net, x)[:, 1])

        dx = [1e-3]
        y = predict_sensitivity(net, x)
        J = y[:, 2:end]
        @test predict(net, x .+ dx) ≈ y[:,1] .+ J * dx
    end
end

@testset "random shapes" begin
    srand(10)
    for i in 1:100
        widths = [rand(1:5) for i in 1:4]
        x = randn(widths[1])
        params = randn(Params{Float64}, widths)
        net = Net(params)
        @test @inferred(predict(net, x)) == @inferred(predict_sensitivity(net, x)[1:widths[end]])
        for i in 1:length(x)
            dx = zeros(x)
            dx[i] = 1e-3
            out = predict_sensitivity(net, x)
            y = out[:, 1]
            J = out[:, 2:end]
            @test predict(net, x .+ dx) ≈ y .+ J * dx
        end
    end
end

@testset "compiled jacobian" begin
    srand(20)
    for i in 1:10
        widths = [rand(1:5) for i in 1:4]
        g = (data, x) -> sum(predict(Net(Params(widths, data)), x))
        data = randn(Params{Float64}, widths).data
        x = randn(widths[1])
        g_tape = ReverseDiff.compile(ReverseDiff.GradientTape(g, (data, x)))
        results = (similar(data), similar(x))

        for i in 1:50
            x = randn(widths[1])
            data = randn(Params{Float64}, widths).data
            ReverseDiff.gradient!(results, g_tape, (data, x))
            @test all(results .≈ ReverseDiff.gradient(g, (data, x)))
        end
    end

    for i in 1:10
        widths = [rand(1:5) for i in 1:4]
        g = (data, x) -> sum(predict_sensitivity(Net(Params(widths, data)), x))
        data = randn(Params{Float64}, widths).data
        x = randn(widths[1])
        g_tape = ReverseDiff.compile(ReverseDiff.GradientTape(g, (data, x)))
        results = (similar(data), similar(x))

        for i in 1:50
            x = randn(widths[1])
            data = randn(Params{Float64}, widths).data
            ReverseDiff.gradient!(results, g_tape, (data, x))
            @test all(results .≈ ReverseDiff.gradient(g, (data, x)))
        end
    end
end

@testset "input output scaling" begin
    srand(1)
    for i in 1:10
        widths = [rand(1:5) for i in 1:4]
        t_in = AffineMap(randn(widths[1], widths[1]), randn(widths[1]))
        t_out = AffineMap(randn(widths[end], widths[end]), randn(widths[end]))
        params = randn(Params{Float64}, widths)
        net = Net(params, t_in, t_out)
        for j in 1:10
            x0 = randn(widths[1])
            yJ = predict_sensitivity(net, x0)
            y = yJ[:, 1]
            J = yJ[:, 2:end]
            @test predict(net, x0) == y

            @test ForwardDiff.jacobian(net, x0) ≈ J
        end
    end
end


