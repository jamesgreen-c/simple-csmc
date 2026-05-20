# JAX Conditional Sequential Monte Carlo Example

This repository provides a small, self-contained example of how to implement and run Conditional Sequential Monte Carlo (cSMC) in JAX.

The code shows the main ingredients needed for a cSMC implementation:

- a state-space model,
- proposal kernels,
- importance weights,
- resampling,
- backward sampling,
- vectorised repeated experiments,
- plotting against the exact Kalman smoother in a linear Gaussian model.

The main example uses a linear Gaussian state-space model (LGSSM). This is useful because the exact smoothing distribution is available from the Kalman smoother, giving a reliable reference against which the particle smoother can be compared.

## Repository structure

A typical layout is:

```text
.
├── example.py              # Runs the cSMC / SMC experiment
├── plotting.py             # Plots saved samples against truth and Kalman smoother
├── results/                # Created automatically when experiments are run
└── src/
    ├── kernels/            # Kernel constructors and KernelType enum
    ├── model.py            # Data generation for the LGSSM
    └── utils/
        ├── common.py       # Utility functions, e.g. Barker move
        ├── kalman.py       # Kalman filtering and smoothing routines
        ├── resamplings.py  # Resampling schemes
        └── csmc.py         # Generic cSMC implementation
```

The main file to inspect is `src/utils/csmc.py`, which contains the generic cSMC implementation. The experiment-specific model, proposal functions, weighting functions, and resampling scheme are constructed elsewhere and then passed into the generic kernel.

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Then install the repository in editable mode:

```bash
pip install -e .
```

## Running the example

An example can be run via `example.py`. By default, it uses a bootstrap cSMC kernel, multinomial resampling, and a backward pass to produce smoothing samples.

Run the default experiment with:

```bash
python example.py
```

The default configuration is:

```text
T = 100       # Number of time steps
D = 1         # Dimension of the latent state and observations
K = 1         # Number of independent datasets / experiments
M = 1         # Number of smoothing samples per dataset
N = 31        # Number of non-reference particles; total particles are N + 1
seed = 1234   # Random number generator seed
```

The total number of particles used internally is `N + 1`. This convention is useful for cSMC because one particle slot may be reserved for the reference trajectory when running in conditional mode.

These parameters can be changed by passing command-line arguments. For example:

```bash
python example.py --M 100 --D 5
```

runs the bootstrap cSMC kernel with 100 independent smoothing samples for a system with latent dimension 5.

Another example is:

```bash
python example.py --T 200 --D 2 --M 50 --N 63 --seed 2026
```

which runs a longer experiment with more particles and a different random seed.

The results of each experiment are saved automatically inside the `results/` directory. The output path records the main experimental settings, so different configurations are stored separately.

## Plotting the results

After running an experiment, the results can be plotted with:

```bash
python plotting.py
```

The plotting script loads the saved results from the corresponding folder in `results/` and produces a figure comparing:

- the true latent path,
- the Kalman smoothing mean,
- the particle smoothing sample,
- the noisy observations.

The plotting arguments must match the arguments used to run `example.py`. For example, if the experiment was run with:

```bash
python example.py --M 100 --D 5
```

then the corresponding plot should be generated with:

```bash
python plotting.py --D 5
```

The argument `--M` is not needed for plotting because the number of saved samples is read directly from the saved data file.

If multiple independent experiments were run using `K > 1`, a specific experiment index can be selected with `--i`. For example:

```bash
python plotting.py --D 5 --i 0
```

plots the first independent dataset.

The resulting plot is saved inside the relevant experiment folder in `results/`.

## What the cSMC kernel does

The cSMC kernel has two main stages.

First, it runs a forward particle filter. At each time step, the algorithm:

1. resamples ancestor indices,
2. propagates particles through the proposal kernel,
3. optionally fixes one particle to a reference trajectory,
4. computes importance weights,
5. normalises the weights.

Second, it reconstructs a full latent trajectory. This can be done in two ways:

1. by tracing ancestors backwards through the stored ancestor array, or
2. by using backward sampling.

Backward sampling is usually preferable because it reduces path degeneracy. In a standard particle filter, many final particles can share the same early ancestors, especially for long time series. Backward sampling partially mitigates this by resampling the previous state conditional on the future selected state, rather than simply following the originally sampled genealogy.

## Conditional and unconditional modes

The same implementation can be used for both ordinary SMC and conditional SMC.

When `conditional=False`, the algorithm runs as an ordinary particle filter / particle smoother. No reference trajectory is fixed.

When `conditional=True`, one particle path is fixed to a supplied reference trajectory `x_star`, with corresponding reference indices `b_star`. This gives a Markov kernel on complete latent trajectories, which is the standard cSMC update used inside Particle Gibbs.

In the example script, the kernel is constructed with:

```python
conditional=False
```

This means the example is being used to draw independent smoothing samples. To demonstrate full conditional SMC, one would repeatedly feed the previous sampled trajectory back into the next kernel call as the reference trajectory.

## Benefits of JAX

This repository uses JAX for three main reasons:

1. Automatic vectorisation with `jax.vmap`,
2. Efficient looping with `jax.lax.scan`,
3. Just-in-time compilation with `jax.jit`.

Together, these allow the particle filter to be written in a way that is close to the mathematical algorithm, while still being compiled into efficient array operations.

### JIT compilation

The experiment function is JIT-compiled with:

```python
@jax.jit
def one_experiment(key):
    ...
```

JIT compilation traces the function once and compiles it into an optimised computation graph. This is useful here because the particle filtering recursion contains many repeated array operations.

### Vectorisation over particles

Particles are stored with the particle index as the leading axis. At a single time step, the particle array has shape:

```text
(N + 1, D)
```

where `N + 1` is the total number of particles and `D` is the state dimension.The proposal function is applied to all particles simultaneously:

```python
x_t = M_t_rvs(key_proposal_t, x_t_m_1, M_t_params)
# x_t_m_1.shape == (N + 1, D)
# x_t.shape     == (N + 1, D)
```

The weights are also computed for all particles at once:

```python
log_w_t = Gamma_t(x_t_m_1, x_t, Gamma_params_t) - M_t_logpdf(x_t_m_1, x_t, M_t_params)
# log_w_t.shape == (N + 1,)
```

This avoids explicit Python loops over particles.

### Scanning over time

The time recursion is handled using `jax.lax.scan`:

```python
_, (log_ws, As, xs) = jax.lax.scan(body, (w0, x0), inputs)
```

This replaces a Python loop over time with a JAX-compatible loop. The main saved arrays have shapes:

```text
xs.shape     == (T, N + 1, D)
log_ws.shape == (T, N + 1)
As.shape     == (T - 1, N + 1)
```

Here:

- `xs` stores the particles at every time step,
- `log_ws` stores the normalised log weights,
- `As` stores the ancestor indices produced by resampling.

Using `lax.scan` allows the full filtering recursion to be compiled by JAX. A normal Python loop would not be compatible with JIT compilation.

### Vectorisation over smoothing samples

In `example.py`, repeated smoothing samples are generated using `jax.vmap`.

The function:

```python
def _smoothing_sample(k_):
    sample, *_ = kernel_(k_, init_state)
    return sample
```

runs one smoothing sample using one random key. Then:

```python
sample_keys = jax.random.split(sample_key, args.M)
samples = jax.vmap(_smoothing_sample)(sample_keys)
```

runs the same function independently over `M` random keys.

The resulting array has shape:

```text
samples.shape == (M, T, D)
```

This means the `M` smoothing samples are vectorised rather than generated sequentially in a Python loop.

### Looping over independent datasets

The outer loop over `K` independent datasets is written as a Python loop:

```python
for k, key_k in enumerate(EXPERIMENT_KEYS):
    samples_k, s_ms_k, s_covs_k, true_xs_k, ys_k = one_experiment(key_k)
```

This keeps progress bars, saving, and debugging straightforward. It is also useful for large experiments, where D, T, or K may be large, because only one dataset is processed at a time. Compared with vectorising over K, this can reduce the peak memory requirement by approximately a factor of K.

In principle, this outer loop could be replaced with jax.vmap. However, doing so would trace and execute all K experiments as one larger computation, which can allocate substantially more memory and become impractical for large systems.

## Saved output

Each experiment saves a compressed NumPy file called `data.npz`. This contains:

```text
smoothing_means  # shape (K, T, D)
smoothing_covs   # shape (K, T, D, D)
samples          # shape (K, M, T, D)
true_xs          # shape (K, T, D)
ys               # shape (K, T, D)
```

These arrays are then loaded by `plotting.py`.

## Summary

This repository is intended as a minimal example of cSMC in JAX. The key implementation ideas are:

- use array operations over particles,
- use `jax.lax.scan` over time,
- use `jax.vmap` over independent smoothing samples,
- optionally use `jax.jit` to compile the full experiment,
- compare against the Kalman smoother in a model where the exact smoothing distribution is known.
