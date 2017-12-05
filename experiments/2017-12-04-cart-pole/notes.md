Trained on cart-pole using MPC optimization:

```
mpc_params = LearningMPC.MPCParams(
    Δt=0.05,
    horizon=10,
    mip_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=120, MIPGap=1e-1, FeasibilityTol=1e-3),
    lcp_solver=GurobiSolver(Gurobi.Env(), OutputFlag=0))
```

Best performance was with gradient sensitivity set to 0.2. Learning rate was fixed at 1e-3. Random states were generated with configuration sigma 0.1 and velocity sigma 0.5. MPC controller was called with probability 0.2 at each DAGGER step. Data was split by trajectory, 50% train, 25% validate, 25% test. 15 rounds of training were performed; each round consisted of 4 trajectories and 5 ADAM updates. 
