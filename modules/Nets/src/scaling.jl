"""
Scale a dataset consisting of state inputs and control+sensitivity outputs. For example, let's say our true function is:

    y1 = 2x1 + x2 + 1
    y2 = x1 + 4x2 - 3

We can put in some input data and see what we get:

    x         y
--------------------
[-1, -1]     [-2, -8]
[0, 0]       [1, -3]
[1, 1]       [4, 2]

For our controls application, we also track the desired sensitivity of
the network (the partial derivatives of y_i w.r.t. x_j. Let's write those
sensitivities into our dataset by adding a second dimension:

    x         y
--------------------
[-1, -1]     [-2  2  1
              -8  1  4]
[0, 0]       [1   2  1
              -3  1  4]
[1, 1]       [4   2  1
              2   1  4]

We can scale the *rows* of y however we want, but we can't scale individual elements. That's because we have to maintain the fact that columns 2 and 3 are actually the jacobian of column 1 w.r.t. the input. 

Let's scale the rows of y so that they're between -1 and 1

    x         y .* [1/4, 1/8]
--------------------
[-1, -1]    [-1/2 1/2  1/4
             -1   1/8  1/2]
[0, 0]      [1/4  1/2  1/4
             -3/8 1/8  1/2]
[1, 1]      [1    1/2  1/4
             1/4  1/8  1/2

v1 = 1/2 x1 + 1/4 x2 + 1/4
v2 = 1/8 x1 + 1/2 x2 - 3/8

But now our gradients are only in the range [0, 1/2]. We'd really like to 
rescale them up to [-1, 1]. We can't apply an offset to the gradients, but we can at least scale the individual components by replacing x with a scaled version. 

  x .* [1/2, 1/2]  y .* [1/4, 1/8]
----------------------------------
[-1/2, -1/2]       [-1/2  1    1/2
                    -1    1/4  1]
[0, 0]             [1/4   1    1/2
                    -3/8  1/4  1]
[1/2, 1/2]         [1     1    1/2
                    1/4   1/4  1]

Let's check the function that we're learning now:

v1 = u1 + 1/2 u2 + 1/4
v2 = 1/4 u1 + u2 - 3/8

Now we have a nice function with outputs and gradients which are within [-1, 1]. We can apply the two transforms above to our training and test data and then try to learn v(u). 

When we actually want to deploy our network, we need the original, un-transformed outputs. To do that, we want to:

* take x0
* apply the x -> u transformation (multiply by [1/2, 1/2])
* run the network to get v
* apply the inverse of the y -> v transformation (multiply by [4, 8])

Let's verify that by hand:

x = [-2, -2]
apply x -> u to get u = [-1, -1]
run the network to get v = [-1.25, -1.625]
apply the inverse of y -> v to get [-5, -13]

Expected result is [-5, -13]. It works!
"""

function _scale(sample, x_to_u, u_to_x, y_to_v)
    x, yJ = sample
    y = yJ[:, 1]
    J = yJ[:, 2:end]
    u = x_to_u(x)
    v = y_to_v(y)
    Jvu = (transform_deriv(u_to_x, x) * (transform_deriv(y_to_v, y) * J)')'
    (u, hcat(v, Jvu))
end

function rescale(data)
    x, yJ = first(data)
    y_scale = zeros(size(yJ, 1))
    J_scale = zeros(size(yJ, 2) - 1)
    for (x, yJ) in data
        y_scale .= max.(y_scale, abs.(yJ[:, 1]))
    end
    for (x, yJ) in data
        J_scale .= max.(J_scale, [maximum(abs, yJ[:, i + 1] ./ y_scale) for i in 1:length(J_scale)])
    end

    v_to_y = AffineMap(diagm(y_scale), zeros(y_scale))
    y_to_v = inv(v_to_y)
    x_to_u = AffineMap(diagm(J_scale), zeros(J_scale))
    u_to_x = inv(x_to_u)

    _scale.(data, x_to_u, u_to_x, y_to_v), x_to_u, v_to_y
end

