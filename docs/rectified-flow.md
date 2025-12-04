# 0. Introduction

This notebook builds a unified understanding of how Rectified Flow (RF) dynamics can be solved efficiently using the Unified Predictor–Corrector (UniPC) framework.

We begin by formulating rectified flow as a simple, deterministic motion in log-SNR space, where data points evolve along straight lines between pure noise and clean signal.

We then derive UniPC from first principles as a general solver for linear inhomogeneous ODEs, introducing its Green’s-function foundation and numerical discretization rules.

Finally, we apply UniPC to the rectified flow equation, showing how the continuous formulation reduces naturally to practical discrete updates, from the zeroth-order explicit Euler form to higher-order predictor–corrector variants.

Together, these sections provide both intuition and mathematical structure for how rectified flow models are integrated in practice, bridging physical dynamics, neural prediction, and numerical ODE solvers under a single unified lens.

# 1. Rectified Flow Formulation (RF)

![flow-matching](https://github.com/user-attachments/assets/6652593d-e044-4234-86b9-288616d30331)

Rectified flow describes a deterministic, straight-line evolution between a clean signal $x_0$ and a noise sample $\epsilon$.
The sample at any intermediate time $t$ is given by

```math
x(t) = \alpha(t) x_{0} + \sigma(t) \epsilon
```

where

* $\alpha(t)$ is the signal amplitude,
* $\sigma(t)$ is the noise amplitude.

Differentiating with respect to $t$,

```math
v(t) \equiv \frac{dx}{dt} = \dot{\alpha}(t) x_{0} + \dot{\sigma}(t) \epsilon
```

For rectified flow, the signal and noise amplitudes satisfy a straight-line constraint:

```math
\alpha(t) + \sigma(t) = 1 \quad\Rightarrow\quad \dot{\alpha}(t) + \dot{\sigma}(t) = 0
```

The interested reader should contrast this to other formulations, say, the variance-preserving formulation, where $\alpha(t)^2 + \sigma(t)^2 = 1$.

We define the log-signal-to-noise ratio (log-SNR) as

```math
\lambda(t) \equiv \log \alpha(t) - \log \sigma(t)
```

so that $\alpha \in [0,1]$ corresponds to $\lambda \in (-\infty, \infty)$. This is just a change of variable that helps simplify the math.

Differentiating,

```math
\dot{\lambda}
= \frac{\dot{\alpha}}{\alpha} - \frac{\dot{\sigma}}{\sigma}
= \frac{\dot{\alpha}}{\alpha} + \frac{\dot{\alpha}}{\sigma}
= \dot{\alpha} \left( \frac{1}{\alpha} + \frac{1}{\sigma} \right)
```

We can now change variables from $t$ to $\lambda$:

```math
\frac{dx}{d\lambda}
= \frac{\frac{dx}{dt}}{\frac{d\lambda}{dt}}
= \frac{\dot{\alpha} x_{0} + \dot{\sigma} \epsilon}
{\dot{\alpha} \left( \frac{1}{\alpha} + \frac{1}{\sigma} \right)}
```

Substituting $x = \alpha x_0 + \sigma \epsilon$ and simplifying 2.by step,

```math
\frac{dx}{d\lambda}
= \frac{\frac{\dot{\alpha}}{\alpha}(x - \sigma \epsilon) + \dot{\sigma} \epsilon}
{\dot{\alpha} \left( \frac{1}{\alpha} + \frac{1}{\sigma} \right)}
= \frac{\frac{1}{\alpha}(x-\sigma\epsilon)-\epsilon}{\frac{1}{\alpha}+\frac{1}{\sigma}}
```

Using the rectified-flow identities, this simplifies to

```math
\frac{dx}{d\lambda} = \sigma (x - \epsilon), \qquad \lambda \in (-\infty, \infty)
```

which equivalently can be written as

```math
\frac{dx}{d\lambda} = - \alpha (x - x_{0}), \qquad \lambda \in (-\infty, \infty)
```

This expresses the rectified flow ODE in log-SNR ($\lambda$) space, where integration proceeds from large negative $\lambda$ (noisy state) to large positive $\lambda$ (clean state).

# 2. Unified Predictor–Corrector (UniPC)

The Unified Predictor–Corrector (UniPC) method builds on the inhomogeneous linear ordinary differential equation (ODE) form often used in diffusion models and other generative solvers. Below we will derive it from first principles in a pedagogical way so that there is no mystery left.

## 2.1. Start from the inhomogeneous velocity equation

```math
\frac{dx(t)}{dt} = A(t)x(t) + b(t)
```

This is a linear time-varying ODE.

* $A(t)$ describes how the system scales or rotates the state $x(t)$.
* $b(t)$ is a forcing term, representing an external contribution driving the evolution (for example, the denoising velocity in diffusion models).

## 2.2. Separate the homogeneous part

We first solve the homogeneous equation:

```math
\frac{dx_h(t)}{dt} = A(t)x_h(t)
```

Divide both sides by $x_h(t)$:

```math
\frac{dx_h(t)}{x_h(t)} = A(t)dt
```

Integrate both sides from $s$ to $t$:

```math
\int_s^t \frac{dx_h(\tau)}{x_h(\tau)} d\tau = \int_s^t A(\tau)d\tau
```

The left-hand side integrates to a logarithmic ratio:

```math
\log\frac{x_h(t)}{x_h(s)} = \int_s^t A(\tau)d\tau
```

Exponentiate both sides:

```math
x_h(t) = \exp\left(\int_s^t A(\tau)d\tau\right) x_h(s)
```

This is the homogeneous solution, which shows how the system evolves without external forcing.

## 2.3. Relax the initial condition

To solve the inhomogeneous equation, introduce a new function $F(t)$:

```math
x(t) = \exp\left(\int_s^t A(\tau)d\tau\right) F(t)
```

where $F(s) = x(s)$.
This substitution lets $F(t)$ “absorb” the effect of the forcing term $b(t)$.

## 2.4. Differentiate and plug into the original equation

Compute the derivative of $x(t)$:

```math
\frac{dx(t)}{dt}
= A(t)\exp\left(\int_s^t A(\tau)d\tau\right)F(t)
 + \exp\left(\int_s^t A(\tau)d\tau\right)\frac{dF(t)}{dt}
```

Substitute into the original ODE:

```math
A(t)\exp\left(\int_s^t A(\tau)d\tau\right)F(t)
 + \exp\left(\int_s^t A(\tau)d\tau\right)\frac{dF(t)}{dt}
= A(t)x(t) + b(t)
```

Since $x(t) = \exp(\int_s^t A)d\tau F(t)$, the $A(t)x(t)$ terms cancel, leaving:

```math
\exp\left(\int_s^t A(\tau)d\tau\right)\frac{dF(t)}{dt} = b(t)
```

## 2.5. Solve for $F(t)$

Rearranging and integrating both sides:

```math
\frac{dF(t)}{dt} = \exp\left(-\int_s^t A(\tau)d\tau\right)b(t)
```

Integrate from $s$ to $t$:

```math
F(t) = x(s) + \int_s^t \exp\left(-\int_s^{t'}A(\tau)d\tau\right) b(t') dt'
```

## 2.6. Substitute back for $x(t)$

Substitute $F(t)$ back into the earlier substitution:

```math
\begin{align*}

x\left(t\right)&=\exp\left(\int_{s}^{t}A\left(\tau\right)d\tau\right)\left[x\left(s\right)+\int_{s}^{t}\exp\left(-\int_{s}^{t'}A\left(\tau\right)d\tau\right)b\left(t'\right)dt'\right] \\

x\left(t\right)&=\exp\left(\int_{s}^{t}A\left(\tau\right)d\tau\right)x\left(s\right)+\int_{s}^{t}\exp\left(\int_{s}^{t}A\left(\tau\right)d\tau-\int_{s}^{t'}A\left(\tau\right)d\tau\right)b\left(t'\right)dt' \\

&=\exp\left(\int_{s}^{t}A\left(\tau\right)d\tau\right)x\left(s\right)+\int_{s}^{t}\exp\left(\int_{t'}^{t}A\left(\tau\right)d\tau\right)b\left(t'\right)dt' \\

\end{align*}
```

## 2.7. Define the Green's function

Define the Green's function (or propagator) $G(t, s)$ so that the general solution becomes compact:

```math
\begin{align*}
G\left(t,s\right)&\equiv\exp\left(\int_{s}^{t}A\left(\tau\right)d\tau\right) \\

\rightarrow x\left(t\right)&=\underbrace{G\left(t,s\right)x\left(s\right)}_{\text{homogeneous solution}}+\underbrace{\int_{s}^{t}G\left(t,t'\right)b\left(t'\right)dt'}_{\text{forcing term solution}}

\end{align*}
```

## 2.8. Numerical Approximation and the UniPC Update Rule

In practice, we only evaluate this at a discrete set of time steps

```math
t_0 < t_1 < \dots < t_N
```

and we want a fast, accurate way to approximate the integral term.

Moreover, so far we assumed $b$ depends only on $t$. We will now relax this and assume that in diffusion models $b=b_\theta(x(t),t)$, predicted by a neural network $\theta$.

### 2.8.1. Discretizing the Green's-function form

Let $x_i = x(t_i)$.
For a small step from $t_{i}$ to $t_{i+1}$,

```math
x_{i+1}
= G(t_{i+1}, t_i) x_i
 + \int_{t_i}^{t_{i+1}} G(t_{i+1}, t') b_\theta(x(t'), t') dt'.
```

In diffusion models, $A(t)$ and $b(t)$ are often simple functions of $t$,
so $G(t_{i+1},t_i)$ can be precomputed analytically, while the integral is approximated numerically.

### 2.8.2. From integral to predictor–corrector form

The goal is to approximate

```math
\int_{t_i}^{t_{i+1}} G(t_{i+1}, t') b_\theta(x(t'), t') dt'
```

using previously evaluated drifts $b_i \equiv b_\theta(x_i, t_i), b_{i-1} \equiv b_\theta(x_{i-1}, t_{i-1}), \dots$

UniPC interprets this as a unified expansion:

```math
x_{i+1}
= G_{i+1,i} x_i
 + h_i \Phi(b_i, b_{i-1}, \dots),
```

where $h_i = t_{i+1}-t_i$ and
$\Phi$ is a predictor–corrector operator that blends explicit and implicit information.

### 2.8.3. Predictor phase (explicit update)

The predictor estimates an intermediate $\tilde{x}_{i+1}$ using an explicit step:

```math
\tilde{x}_{i+1}
= G_{i+1,i} x_i
 + h_i \sum_{k=1}^{p} \beta_k G_{i+1,i+1-k} b_{i+1-k},
```

where the coefficients $\beta_k$ are chosen so the local truncation error is $O(h_i^{p+1})$.
This step is fast because it only uses already-known drifts.

### 2.8.4. Corrector phase (implicit refinement)

After predicting $\tilde{x}_ {i+1}$, we evaluate the model again to get $b(\tilde{x}_ {i+1}, t_{i+1})$.
Then we correct the trajectory using a trapezoidal- or Adams–Moulton-like term:

```math
\begin{align*}
x_{i+1} = G_{i+1,i} x_i + h_i \sum_{k=1}^{p} \alpha_k G_{i+1,i+1-k} b(x_{i+1-k}, t_{i+1-k}) + h_i \alpha_0 G_{i+1,i+1} b(\tilde{x}_{i+1},t_{i+1})
\end{align*}
```

where the the last term now includes the newest velocity estimate. This provides higher-order accuracy and stability, especially for stiff ODEs. Note that by definition $G_{i+1,i+1} = I$, therefore we could have skipped writing it.

# 3. Solution to Rectified Flow (RF) Using UniPC

Below we connect the last two sections and first arrive at the continuous expression for the RF formulation using UniPC and then derive the discrefe form.

## 3.1 The continuous form

Starting from the rectified-flow ODE derived at the end of section 1.,

```math
\frac{dx}{d\lambda} = \sigma (x - \epsilon), \qquad \lambda \in (-\infty, \infty)
```

the general UniPC solution form is

```math
x(\lambda) = G(\lambda,s) x(s)
 + \int_{s}^{\lambda} G(\lambda,\tau) b(\tau) d\tau
```

We identify the variable transformation

```math
\lambda \equiv \log \frac{\alpha}{\sigma}
```

which implies

```math
\exp(\lambda) = \frac{1 - \sigma}{\sigma}
\quad\Rightarrow\quad
\exp(\lambda) + 1 = \frac{1}{\sigma}
\quad\Rightarrow\quad
\sigma = \frac{1}{\exp(\lambda) + 1}
```

The Green’s function derived in Section 2.7. becomes

```math
G(\lambda,s)
= \exp \left( \int_{s}^{\lambda} \sigma(\tau) d\tau \right)
= \exp \left( \int_{s}^{\lambda} \frac{1}{\exp(\tau) + 1} d\tau \right)
= \exp \left( \log \left| \frac{\exp(\tau)}{\exp(\tau) + 1} \right|_{s}^{\lambda} \right)
```

which evaluates to

```math
G(\lambda,s)
= \frac{\exp(\lambda)(\exp(s) + 1)}
{\exp(s)(\exp(\lambda) + 1)}
= \frac{\frac{\alpha_{\lambda}}{\sigma_{\lambda}} \left( \frac{\alpha_{s}}{\sigma_{s}} + 1 \right)}
{\frac{\alpha_{s}}{\sigma_{s}} \left( \frac{\alpha_{\lambda}}{\sigma_{\lambda}} + 1 \right)}
= \frac{\frac{\alpha_{\lambda}}{\sigma_{\lambda}} \frac{1}{\sigma_{s}}}
{\frac{\alpha_{s}}{\sigma_{s}} \frac{1}{\sigma_{\lambda}}}
```

and simplifies neatly to

```math
G(\lambda,s) = \frac{\alpha_{\lambda}}{\alpha_{s}}
```

The interested reader can verify that the integration factor is identical in form to that appearing in variance-preserving flow matching.

Substituting this $G$ back into the UniPC integral solution gives

```math
x(\lambda)
= \frac{\alpha_{\lambda}}{\alpha_{s}} x(s)
 + \int_{s}^{\lambda} \frac{\alpha_{\lambda}}{\alpha_{\tau}} \sigma_{\tau} \epsilon d\tau
```

Alternatively, if we formulate it in terms of $x_{0}$ rather than $\epsilon$:

```math
x(\lambda)
= \frac{\sigma_{\lambda}}{\sigma_{s}} x(s)
 + \int_{s}^{\lambda} \frac{\sigma_{\lambda}}{\sigma_{\tau}} \alpha_{\tau} x_{0} d\tau
```

This expression connects the rectified-flow dynamics in log-SNR space to the UniPC discretization framework,
showing that UniPC naturally integrates the straight-line flow between data and noise through the Green’s-function propagation factor $G(\lambda,s)$.

## 3.2 Zeroth-Order (Explicit Euler) Discretization

Taking the zeroth-order approximation of the integral term gives

```math
x(\lambda)
= \frac{\sigma_{\lambda}}{\sigma_{s}}x(s)
+ \sigma_{\lambda}\int_{s}^{\lambda}\frac{\alpha_{\tau}}{\sigma_{\tau}}x_{0}d\tau
```

Keep the perspective that $x_{0} \equiv x_{0,\theta}(x(\lambda), \lambda)$ is the output of the neural network: given a dirty signal $x(\lambda)$ and its relative noise level $x(\lambda)$, it computes the best estimate of the corresponding clean signal.

Now we take the zeroth-order approximation that $x_{0}$ does not change much within the small interval, so it can be taken outside the integral.

```math
x(\lambda)
= \frac{\sigma_{\lambda}}{\sigma_{s}}x(s)
+ \sigma_{\lambda}x_{0}\int_{s}^{\lambda}\exp(\tau)d\tau
= \frac{\sigma_{\lambda}}{\sigma_{s}}x(s)
+ \sigma_{\lambda}x_{0}\left(\exp(\lambda)-\exp(s)\right)
```

If we denote the step size as $h=\lambda-s$, then

```math
x(\lambda)
= \frac{\sigma_{\lambda}}{\sigma_{s}}x(s)
- \sigma_{\lambda}x_{0}\exp(\lambda)\left(\exp(-h)-1\right)
= \frac{\sigma_{\lambda}}{\sigma_{s}}x(s)
- x_{0}\alpha_{\lambda}\left(\exp(-h)-1\right)
```

This provides the explicit-Euler (zeroth-order) update rule, which higher-order UniPC steps refine for improved accuracy and stability.

See function `multistep_uni_p_bh_update` in [`fm_solvers_unipc.py`](../cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py).
