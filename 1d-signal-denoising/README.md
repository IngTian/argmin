# 1-D Signal Denoising via PDHG (Primal-Dual Hybrid Gradient)

This project recovers a **piecewise-constant signal** from a noisy observation
using **Total Variation (TV) regularisation**, solved with the
**Primal-Dual Hybrid Gradient (PDHG)** algorithm (Chambolle & Pock, 2011).

---

## Table of Contents

1. [Problem Setup](#1-problem-setup)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Primal-Dual Reformulation](#3-primal-dual-reformulation)
4. [Proximal Operators — Derivations](#4-proximal-operators--derivations)
5. [The PDHG Algorithm](#5-the-pdhg-algorithm)
6. [Convergence Condition](#6-convergence-condition)
7. [Implementation Notes](#7-implementation-notes)
8. [Running the Code](#8-running-the-code)

---

## 1. Problem Setup

We have a **ground-truth** piecewise-constant signal $x^* \in \mathbb{R}^n$
and observe a noisy version:

$$g = x^* + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2 I)$$

**Goal:** Recover $x^*$ from $g$ by exploiting the fact that $x^*$ is
piecewise-constant — i.e.\ its discrete gradient is *sparse*.

---

## 2. Mathematical Formulation

We solve the **Total Variation (TV) denoising** problem (also known as the
Rudin–Osher–Fatemi model adapted to 1-D):

$$\min_{x \in \mathbb{R}^n} \; \underbrace{\|Kx\|_1}_{F(Kx)\;\text{(TV penalty)}} \;+\; \underbrace{\frac{\lambda}{2}\|x - g\|_2^2}_{G(x)\;\text{(data fidelity)}}$$

where:

| Symbol                              | Meaning                                                   |
|-------------------------------------|-----------------------------------------------------------|
| $x \in \mathbb{R}^n$                | Signal to recover (primal variable)                       |
| $g \in \mathbb{R}^n$                | Noisy observation                                         |
| $K \in \mathbb{R}^{(n-1) \times n}$ | First-order finite-difference matrix                      |
| $\lambda > 0$                       | Regularisation weight (balances fidelity vs.\ smoothness) |

The **finite-difference operator** $K$ is:

$$(Kx)_i = x_{i+1} - x_i, \quad i = 1, \dots, n-1$$

so $\|Kx\|_1 = \sum_i |x_{i+1} - x_i|$ is the **discrete total variation**
of $x$.

---

## 3. Primal-Dual Reformulation

The $\ell_1$ norm is non-smooth, which makes the problem hard to solve with
plain gradient descent. PDHG handles this by rewriting the problem using the
**convex conjugate** (Fenchel conjugate) of $F$.

### Convex conjugate of $F(u) = \|u\|_1$

By definition:

$$F^*(a) = \sup_{u}\; \langle a, u \rangle - \|u\|_1$$

This evaluates to the **indicator function of the $\ell_\infty$ unit ball**:

$$F^*(a) = \begin{cases} 0 & \text{if } \|a\|_\infty \le 1 \\\ +\infty & \text{otherwise} \end{cases}$$

Using the identity $F(u) = \sup_{a}\; \langle a, u \rangle - F^*(a)$, the
original problem becomes the **saddle-point (min-max) problem**:

$$\min_{x} \max_{a} \; \langle a,\, Kx \rangle \;-\; F^*(a) \;+\; G(x)$$

where:

- $x \in \mathbb{R}^n$ is the **primal** variable (the signal),
- $a \in \mathbb{R}^{n-1}$ is the **dual** variable.

---

## 4. Proximal Operators — Derivations

PDHG requires the **proximal operators** of $G$ and $F^*$.

### 4.1 Proximal operator of $\tau G$

$$\mathrm{prox}_{\tau G}(v) = \arg\min_{x} \;\frac{1}{2}\|x - v\|^2 + \tau \cdot \frac{\lambda}{2}\|x - g\|^2$$

Setting the gradient to zero:

$$x - v + \tau\lambda(x - g) = 0$$

$$(1 + \tau\lambda)\,x = v + \tau\lambda\, g$$

$$\boxed{\mathrm{prox}_{\tau G}(v) = \frac{v + \tau\lambda\, g}{1 + \tau\lambda}}$$

**Interpretation:** A weighted average between the current iterate $v$ and the
noisy observation $g$.

### 4.2 Proximal operator of $\sigma F^*$

Since $F^*$ is the indicator of the $\ell_\infty$ unit ball, its proximal
operator is **projection onto that ball**:

$$\mathrm{prox}_{\sigma F^*}(b) = \arg\min_{a:\,\|a\|_\infty \le 1} \;\frac{1}{2}\|a - b\|^2$$

The projection is applied element-wise:

$$\boxed{\mathrm{prox}_{\sigma F^*}(b)_i = \frac{b_i}{\max(1,\; |b_i|)}}$$

**Interpretation:** If a component already satisfies $|b_i| \le 1$ it is left
unchanged; otherwise it is clipped to $\pm 1$.

> **Note:** Because $F^*$ is an indicator function its proximal operator does
> *not* depend on $\sigma$. However, $\sigma$ still appears in the algorithm
> because it scales the argument *before* the proximal step.

---

## 5. The PDHG Algorithm

Initialise:

$$x^0 = g, \quad \bar{x}^0 = g, \quad a^0 = 0$$

Then iterate for $k = 0, 1, 2, \dots$:

$$
\begin{aligned}
a^{k+1} &= \mathrm{prox}_{\sigma F^*}\!\Big(a^k + \sigma\, K\,\bar{x}^k\Big)          &\text{(dual update)} \\[6pt]
x^{k+1} &= \mathrm{prox}_{\tau G}\!\Big(x^k - \tau\, K^\top a^{k+1}\Big)              &\text{(primal update)} \\[6pt]
\bar{x}^{k+1} &= 2\,x^{k+1} - x^k &\text{(extrapolation)}
\end{aligned}
$$

The **extrapolation** step ($\bar{x}$) is key — it accelerates convergence and
is what distinguishes PDHG from a naïve primal-dual method.

---

## 6. Convergence Condition

PDHG converges when the step sizes satisfy:

$$\tau \cdot \sigma \cdot \|K\|^2 < 1$$

For the 1-D finite-difference matrix, the operator norm satisfies
$\|K\|^2 < 4$ (the largest eigenvalue of $K^\top K$ is
$4\sin^2\!\big(\frac{(n-1)\pi}{2n}\big) < 4$).

Therefore a safe choice is:

$$\tau \cdot \sigma < \frac{1}{4} = 0.25$$

In the code we use $\tau = \sigma = 0.499$, giving
$\tau\sigma \approx 0.249 < 0.25$. ✓

---

## 7. Implementation Notes

| File      | Description                                                   |
|-----------|---------------------------------------------------------------|
| `main.py` | Full implementation: signal generation, PDHG solver, plotting |

**Key functions:**

- **`get_K(n)`** — builds the $(n{-}1) \times n$ finite-difference matrix.
- **`prox_tau_G(x, g, tau, lam)`** — proximal operator of the data-fidelity
  term (closed-form weighted average).
- **`prox_sigma_F_star(a)`** — proximal operator of the TV conjugate
  (element-wise projection onto $[-1, 1]$).
- **`pdhg(g, tau, sigma, lam, iter_cnt)`** — main PDHG loop returning the
  denoised signal.

**Hyperparameters in `main.py`:**

| Parameter  | Value | Role                  |
|------------|-------|-----------------------|
| `tau`      | 0.499 | Primal step size      |
| `sigma`    | 0.499 | Dual step size        |
| `lam`      | 0.5   | Regularisation weight |
| `iter_cnt` | 1000  | Number of iterations  |

---

## 8. Running the Code

```bash
pip install -r ../requirements.txt   # numpy, matplotlib
python main.py
```

A plot will appear showing the original, noisy, and denoised signals.

---

## References

- Chambolle, A. & Pock, T. (2011). *A first-order primal-dual algorithm for
  convex problems with applications to imaging.* Journal of Mathematical
  Imaging and Vision, 40(1), 120–145.
- Rudin, L., Osher, S. & Fatemi, E. (1992). *Nonlinear total variation based
  noise removal algorithms.* Physica D, 60(1–4), 259–268.

