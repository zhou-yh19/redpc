## Test the DRS algorithm for solving the L1/L2 optimization problem
using JuMP, MosekTools

function JuMP_solve(D1, D2, G, W, T)
    """
    Solves the given optimization problem through JuMP.
    minimize    ||D1 z||_1 + ||D2 z||_2^2 + 0.5 * ||T - τ||_2^2
    subject to  Gz + Wτ = 0
    # Arguments
    - `D1::Matrix`: The D1 matrix.
    - `D2::Matrix`: The D2 matrix.
    - `G::Matrix`: The G matrix.
    - `W::Matrix`: The W matrix.
    - `T::Vector`: The T vector.
    # Returns
    - `z_opt::Vector`: Optimal solution for z.
    - `tau_opt::Vector`: Optimal solution for τ.
    """
    _, n = size(G)
    _, m = size(W)
    model = Model(MosekTools.Optimizer)
    set_silent(model)
    
    @variable(model, z[1:n])
    @variable(model, τ[1:m])
    @variable(model, t>=0.0)

    # Objective: Minimize ||D1 z||_1 + ||D2 z||_2^2 + 0.5 * ||T - τ||_2^2
    @objective(model, Min, t + sum((D2 * z).^2) + 0.5 * sum((T - τ).^2))
    @constraint(model, [t; D1 * z] in MOI.NormOneCone(1 + n))
    @constraint(model, G * z + W * τ .== 0)

    optimize!(model)

    z_opt = value.(z)
    τ_opt = value.(τ)
    
    return z_opt, τ_opt
end

function soft_thresholding(x, α, D1, D2)
    n = length(x)
    sh_x = zeros(n)
    for i = 1:n
        if x[i] - α * abs(D1[i,i]) > 0
            sh_x[i] = (x[i] - α * abs(D1[i,i])) / (1 + 2α * (D2[i,i])^2)
        elseif x[i] + α * abs(D1[i,i]) < 0
            sh_x[i] = (x[i] + α * abs(D1[i,i])) / (1 + 2α * (D2[i,i])^2)
        else
            sh_x[i] = 0
        end
    end
    return sh_x
end

function DRS_solve(G, W, T, D1, D2, α, max_iters)
    """
    Solves the given optimization problem through the DRS algorithm.
    minimize    ||D1 z||_1 + ||D2 z||_2^2 + 0.5 * ||T - τ||_2^2
    subject to  Gz + Wτ = 0
    # Arguments
    - `G::Matrix`: The G matrix.
    - `W::Matrix`: The W matrix.
    - `T::Vector`: The T vector.
    - `D1::Matrix`: The D1 matrix.
    - `D2::Matrix`: The D2 matrix.
    - `α::Float64`: The step size.
    - `max_iters::Int`: The maximum number of iterations.
    # Returns
    - `z::Vector`: Optimal solution for z.
    - `τ::Vector`: Optimal solution for τ.
    """
    _, n = size(G)
    _, m = size(W)
    z = zeros(n)
    τ = zeros(m)
    ξ = zeros(n)
    η = zeros(m)
    sh = zeros(n)  # Placeholder for sh(ξ^k)
    
    # Extended matrix G~
    G_ext = [G W]

    # Precompute (I - G~⁺G~)
    I_GG = I(n+m) - pinv(G_ext) * G_ext

    for k = 1:max_iters
        # Update sh(ξ^k) based on the soft-thresholding operation
        sh = soft_thresholding(ξ, α, D1, D2)
        
        # DRS update for z^(k+1) and τ^(k+1)
        temp = I_GG * [2 * sh - ξ; T]
        z = temp[1:n]
        τ = temp[n+1:end]

        # Update ξ^(k+1) and η^(k+1)
        ξ = ξ + z - sh
        η = η + τ - (T + η) / 2
    end

    # The final z and τ are given by z_next and τ_next after the last iteration
    return z, τ
end

## Check python executable path of PyCall
using PyCall
@pyimport sys
@show sys.executable
# Try to import torch and check whether cuda is available
@pyimport torch
@pyimport numpy as np
@show torch.cuda.is_available()
function Torch_solve(G, W, T, D1, D2, bs, alpha, iters)
    """
    Solves the given optimization problem through the DRS algorithm implemented in PyTorch.
    minimize    ||D1 z||_1 + ||D2 z||_2^2 + 0.5 * ||T - τ||_2^2
    subject to  Gz + Wτ = 0
    # Arguments
    - `G::Matrix`: The G matrix.
    - `W::Matrix`: The W matrix.
    - `T::Vector`: The T vector.
    - `D1::Vector`: The D1 vector.
    - `D2::Vector`: The D2 vector.
    - `bs::Int`: The batch size.
    - `alpha::Float64`: The step size.
    - `iters::Int`: The maximum number of iterations.
    # Returns
    - `z::Vector`: Optimal solution for z.
    - `τ::Vector`: Optimal solution for τ.
    """
    # Initialize the pytorch solver available at src/modules/l1_minimization_solver.py
    py"""
    import sys
    sys.path.append('.')
    from src.modules.l12_prox_solver import L12ProxSolver
    """
    L12Solver = py"L12ProxSolver"
    # Cast A, b to torch tensors and append singleton batch dimension
    G_torch = torch.tensor(G, dtype=torch.float64, device="cuda:0").unsqueeze(0)
    W_torch = torch.tensor(W, dtype=torch.float64, device="cuda:0").unsqueeze(0)
    T_torch = torch.tensor(T, dtype=torch.float64, device="cuda:0").unsqueeze(0)
    D1_torch = torch.tensor(D1, dtype=torch.float64, device="cuda:0")
    D2_torch = torch.tensor(D2, dtype=torch.float64, device="cuda:0")
    # Solve problem
    solver = L12Solver("cuda:0")
    solution_z, solution_tau = solver(G_torch, W_torch, T_torch, D1_torch, D2_torch, bs, alpha, iters)
    # Cast back to julia array
    solution_z = convert(Array, solution_z.squeeze(0).cpu().numpy())
    solution_tau = convert(Array, solution_tau.squeeze(0).cpu().numpy())
    return solution_z, solution_tau
end

## Test the DRS algorithm for solving the L1/L2 optimization problem
m=200
n=371

# load from numpy
λ_g1 = 0.5
λ_g2 = 30.0
λ_y1 = 100000
λ_y2 = 100000

A_aug = [zeros(20, 20); -Matrix{Float64}(I, 20, 20); zeros(80, 20); zeros(80, 20)]

D1_vec = [λ_g1 * ones(n); λ_y1 * ones(20)]
D2_vec = [sqrt(λ_g2) * ones(n); sqrt(λ_y2) * ones(20)]
D1 = Diagonal(D1_vec)
D2 = Diagonal(D2_vec)

##
for i in 1:10
    G = randn(m, n)
    G = [G A_aug]
    T = b_list[i,:]
    W = Matrix{Float64}(I, m, m)

    zj, τj = JuMP_solve(D1, D2, G, W, T)
    zp, τp = DRS_solve(G, W, T, D1, D2, 1.0, 500)
    zt, τt = Torch_solve(G, W, T, D1_vec, D2_vec, 1, 1.0, 500)

    sj = sum(abs.(D1 * zj)) + sum((D2 * zj).^2) + 0.5 * sum((T - τj).^2)
    sp = sum(abs.(D1 * zp)) + sum((D2 * zp).^2) + 0.5 * sum((T - τp).^2)
    st = sum(abs.(D1 * zt)) + sum((D2 * zt).^2) + 0.5 * sum((T - τt).^2)

    @show sj, sp, st
    @show norm(zj, 1), norm(zp, 1), norm(zt, 1)
    @show norm(zj-zp)
    @show norm(τj-τp)
    @show norm(zj-zt)
    @show norm(τj-τt)
end