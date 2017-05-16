# Explicit MPC and Affine Neural Nets

A non-hybrid QP has a piecewise affine MPC solution (see the multiparametric QP work), and a neural network with ReLU activations also has a piecewise affine output. So we should be able to train a NN to reproduce the piecewise affine solution. Results are in the `explicit-mpc.ipynb` notebook. 

System was the double integrator: 

mass = 1.
l = 1.
g = 10.
N = 4
A = [0. 1.;
     g/l 0.]
B = [0 1/(mass*l^2.)]'
Δt = .1

I gathered data by running 10,000 MPC problems (using Gurobi and JuMP) and split into 60% training, 40% test. NN structure was:

mlp = @mx.chain mx.Variable(:x) => 
mx.FullyConnected(name=:fc1, num_hidden=4) =>
mx.Activation(name=:relu1, act_type=:relu) =>
mx.FullyConnected(name=:fc2, num_hidden=4) =>
mx.Activation(name=:relu2, act_type=:relu) => 
mx.FullyConnected(name=:fc3, num_hidden=4) => 
mx.Activation(name=:relu3, act_type=:relu) =>
mx.FullyConnected(name=:fc4, num_hidden=1) =>
mx.LinearRegressionOutput(name=:u)

Trained with ADAM (learning rate = 0.1) for 20 epochs. Batch size was 20. Weights initialized randomly between -1 and 1. 

Policy: `policy.svg`. White lines are the decision boundaries (see `explicit_network.jl`). 

I then ran the explicit MPC tool from Tobia on that same system. I plotted the decision boundaries of the network on top of his regions (his regions are in color and are colored by their first control input). They agree quite well!