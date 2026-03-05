import matplotlib.pyplot as plt
import numpy as np


# ===========================================================================
# Proximal Operators
# ===========================================================================
#
# In PDHG we need two proximal operators:
#   1. prox of the DATA-FIDELITY term   G(x) = (lam/2)||x - g||^2
#   2. prox of the CONJUGATE of the TV  F*(a) = indicator_{||.||_inf <= 1}
#
# Each one has a clean closed-form solution derived below.
# ===========================================================================

def prox_tau_G(x: np.ndarray, g: np.ndarray, tau: float, lam: float) -> np.ndarray:
    """
    Proximal operator of tau * G, where G(x) = (lam / 2) * ||x - g||^2.

    Derivation (set gradient of the proximal sub-problem to zero):
        prox_{tau*G}(v) = argmin_x  (1/2)||x - v||^2 + tau*(lam/2)||x - g||^2
        Gradient:  (x - v) + tau*lam*(x - g) = 0
        Solve:     x*(1 + tau*lam) = v + tau*lam*g

    Closed-form solution:
        prox_{tau*G}(v) = (v + tau * lam * g) / (1 + tau * lam)

    Interpretation: a weighted average between the current iterate v and
    the noisy observation g.  When tau*lam is large the result is pulled
    closer to g (more faith in the data); when tau*lam is small it stays
    closer to v (more faith in the regulariser).
    """
    # Element-wise weighted average between x (current iterate) and g (data)
    return (x + tau * lam * g) / (1 + tau * lam)


def prox_sigma_F_star(a: np.ndarray) -> np.ndarray:
    """
    Proximal operator of sigma * F*, where F*(a) is the convex conjugate
    of F(u) = ||u||_1.

    Derivation:
        F(u)  = ||u||_1
        F*(a) = sup_u { <a,u> - ||u||_1 }
              = 0        if ||a||_inf <= 1
              = +infty   otherwise       (indicator of the l_inf unit ball)

    Because F* is an indicator function, its proximal operator is just the
    Euclidean PROJECTION onto the l_inf unit ball:
        prox_{sigma*F*}(b) = argmin_{||a||_inf<=1}  (1/2)||a - b||^2

    Element-wise this is simple clipping:
        prox_{sigma*F*}(b)_i = b_i / max(1, |b_i|)

    Note: the result does NOT depend on sigma; sigma only affects how the
    argument is constructed *before* calling this function.
    """
    # Element-wise: clip each component to [-1, 1]
    # Equivalent to: np.clip(a, -1, 1) when |a|>=1, identity otherwise
    return a / np.maximum(1, np.abs(a))


# ===========================================================================
# Finite-Difference Operator  (the "K" in  min_x F(Kx) + G(x))
# ===========================================================================
#
# The total variation of a 1-D signal x is  TV(x) = sum_i |x_{i+1} - x_i|.
# We can write this compactly as  TV(x) = ||Kx||_1  where K is the
# first-order forward-difference matrix:
#
#       K = [ -1  1  0  0 ... 0  0 ]
#           [  0 -1  1  0 ... 0  0 ]     shape: (n-1) x n
#           [        ...            ]
#           [  0  0  0  0 ...-1  1 ]
#
# K^T K is a tridiagonal matrix whose largest eigenvalue is < 4.
# This fact is used to set the PDHG step sizes (see convergence condition).
# ===========================================================================

def get_K(n: int) -> np.ndarray:
    """
    Construct the first-order finite-difference matrix K of shape (n-1, n).

    K computes forward differences: (Kx)_i = x_{i+1} - x_i.

    This is the discrete analogue of the gradient operator, used to measure
    the total variation of a 1-D signal.
    """
    # np.eye(n-1, n, 1) has 1 on the first super-diagonal  (the x_{i+1} part)
    # np.eye(n-1, n, 0) has 1 on the main diagonal          (the x_i part)
    return np.eye(n - 1, n, 1) - np.eye(n - 1, n, 0)


# ===========================================================================
# PDHG Algorithm  (Chambolle–Pock, 2011)
# ===========================================================================
#
# We solve the saddle-point form:
#     min_x  max_a   <a, Kx>  -  F*(a)  +  G(x)
#
# where F*(a) is the indicator of the l_inf unit ball and G(x) is the
# data fidelity.  The algorithm alternates:
#   1. Dual   ascent  on a  (+ prox of F*)
#   2. Primal descent on x  (+ prox of G)
#   3. Extrapolation of x_bar = 2*x_new - x_old  (accelerates convergence)
# ===========================================================================

def pdhg(g: np.ndarray, tau: float, sigma: float, lam: float, iter_cnt: int) -> np.ndarray:
    """
    Primal-Dual Hybrid Gradient (PDHG) method for 1-D total-variation denoising.

    We solve:
        min_x  F(Kx) + G(x)
    where
        F(u)  = ||u||_1            (total variation penalty)
        G(x)  = (lam/2)||x - g||^2 (data fidelity term)
        K     = finite-difference operator

    PDHG iterates (Chambolle–Pock, 2011):
        a^{k+1}     = prox_{sigma * F*}( a^k + sigma * K * x_bar^k )
        x^{k+1}     = prox_{tau * G}  ( x^k - tau * K^T * a^{k+1} )
        x_bar^{k+1} = 2 * x^{k+1} - x^k

    Convergence is guaranteed when tau * sigma * ||K||^2 < 1.
    For the 1-D difference operator, ||K||^2 < 4, so tau * sigma < 0.25
    is sufficient (here we use tau = sigma = 0.499 => product ≈ 0.249).

    Parameters
    ----------
    g        : noisy input signal (length n)
    tau      : primal step size
    sigma    : dual step size
    lam      : regularisation weight (larger => closer to noisy signal)
    iter_cnt : number of PDHG iterations

    Returns
    -------
    x : denoised signal (length n)
    """
    K = get_K(len(g))

    # Dual variable (lives in R^{n-1}, one per finite difference)
    a = np.zeros(len(g) - 1)

    # Primal variable and its extrapolated version, initialised to noisy signal
    x = g
    x_bar = g

    for _ in range(iter_cnt):
        # --- Dual update: gradient ascent on a, then prox of F* ---
        a = prox_sigma_F_star(a + sigma * K @ x_bar)

        # --- Primal update: gradient descent on x, then prox of G ---
        x_tmp = x
        x = prox_tau_G(x - tau * K.T @ a, g, tau, lam)

        # --- Over-relaxation (extrapolation) step ---
        x_bar = 2 * x - x_tmp

    return x


# ---------------------------------------------------------------------------
# Main: generate test signal, run PDHG, and plot results
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # -- 1. Build a piecewise-constant ground-truth signal --
    n = 200
    signal = np.zeros(n)
    signal[:50] = 1
    signal[50:100] = 3
    signal[100:150] = 2
    signal[150:200] = -1

    # -- 2. Corrupt with additive Gaussian noise --
    noise_std = 0.3
    noisy_signal = signal + np.random.normal(0, noise_std, n)

    # -- 3. PDHG hyper-parameters --
    tau = 0.499  # primal step size
    sigma = 0.499  # dual step size  (tau * sigma * ||K||^2 < 1 required)
    lam = 0.5  # regularization weight
    iter_cnt = 1000  # number of iterations

    # -- 4. Run PDHG denoising --
    denoised_signal = pdhg(noisy_signal, tau, sigma, lam, iter_cnt)

    # -- 5. Visualise results --
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Original Signal', linewidth=2)
    plt.plot(noisy_signal, label='Noisy Signal', alpha=0.5)
    plt.plot(denoised_signal, label='Denoised Signal (PDHG)', linewidth=2)
    plt.legend()
    plt.title('1D Signal Denoising using PDHG')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid()
    plt.show()
