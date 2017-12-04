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

@testset "sensitivity gradients" begin
    srand(50)
    for (activation, atol) in [(Nets.leaky_relu, 1e-9), (Nets.hat_relu, 1e-9), (Nets.gaussian, 1e-9)]
        for i in 1:20
            widths = [rand(1:5) for i in 1:4]
            g = (data, x) -> sum(Nets.predict_sensitivity(Nets.Net(Nets.Params(widths, data), activation), x)[:, 2:end])
            data = randn(Nets.Params{Float64}, widths).data
            x = randn(widths[1])

            results = (similar(data), similar(x))
            ReverseDiff.gradient!(results, g, (data, x))
            J = results[1]

            y = g(data, x)

            for j in 1:length(data)
                Δ = zeros(data)
                Δ[j] = 1e-5
                y2 = g(data .+ Δ, x)
                if !isapprox(y2, y + J' * Δ, atol=atol)
                    @show y2 y2 - (y + J' * Δ)
                end
                @test isapprox(y2, y + J' * Δ, atol=atol)
            end
        end
    end
end

@testset "input output scaling" begin
    srand(60)
    for activation in [Nets.leaky_relu, Nets.hat_relu, Nets.gaussian, Nets.elu]
        for i in 1:10
            widths = [rand(1:5) for i in 1:4]
            t_in = AffineMap(randn(widths[1], widths[1]), randn(widths[1]))
            t_out = AffineMap(randn(widths[end], widths[end]), randn(widths[end]))
            params = randn(Params{Float64}, widths)
            net = Net(params, activation, t_in, t_out)
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
end

@testset "scaling" begin
    srand(1)
    f(x) = [2 * x[1] + x[2] + 1, x[1] + 4 * x[2] - 3]

    X = [randn(2) for i in 1:100]
    Y = [hcat(f(x), ForwardDiff.jacobian(f, x)) for x in X];

    train_data = collect(zip(X, Y))

    train_data_scaled, x_to_u, v_to_y = Nets.rescale(train_data)
    for (x, yJ) in train_data
        y = yJ[:, 1]
        J = yJ[:, 2:end]
        @test y ≈ f(x)
        @test J ≈ ForwardDiff.jacobian(f, x)
    end

    u_to_x = inv(x_to_u)
    y_to_v = inv(v_to_y)

    for (u, vJ) in train_data_scaled
        v = vJ[:, 1]
        J = vJ[:, 2:end]
        @test v |> v_to_y ≈ u |> u_to_x |> f
        @test J ≈ ForwardDiff.jacobian(u -> u |> u_to_x |> f |> y_to_v, u)
        @test maximum(abs, J) ≈ 1.0
    end
end

