# optimal_para.py
import numpy as np
import torch
# To obtain the optimal alpha for PASA algorithm using fixed-point iteration.
# The Nonlinear Equation: beta / (1-beta) = f(beta), f(beta) = b*n / (a*(a-b*n))
def obtainInvPam(beta0, N, tp = torch.float16, cp = torch.float64):
    M0 = torch.tensor(1.0, dtype = cp) - beta0.type(cp) / N
    M1 = -beta0.type(cp) / N
    M0 = M0.type(tp)
    M1 = M1.type(tp)
    b = -M1.type(cp)
    a = M0.type(cp) + b
    Inv_Pam = b * N / (a * (a - b * N)) + (1 - a) / a
    return Inv_Pam
def optimal_beta(beta0, N, tol = 1.0e-8, tp = torch.float16, cp = torch.float64):
    err = 1.0
    iter = 0
    Inv_Pam = obtainInvPam(beta0, N, tp, cp)
    while (err > tol):
        Inv_Pam = obtainInvPam(beta0, N, tp, cp)
        beta = Inv_Pam / (1.0 + Inv_Pam)
        err = torch.abs(beta - beta0) / torch.abs(beta0)
        beta0 = beta * 1.0
        iter += 1
    return beta
if __name__ == "__main__":
    # float16
    print("=======================float16(1,5,10)==================")
    print("Initial beta = 1-1/2**4, 1-1/2**5, 1-1/2**6")
    bits = int(3)
    beta0 = torch.zeros(bits)
    for i in range(bits):
        beta0[i] = 1.0 - 1.0 / 2**(i+4)
    N = int(128)
    M = beta0.size()
    M = M[0]
    beta = torch.zeros(M)
    for i in range(M):
        beta[i] = optimal_beta(beta0[i], N, tp = torch.float16)
    print("========================Results:=========================")
    print(f"for float16, initial beta: {beta0}")
    print(f"for float16, beta: {beta}")
