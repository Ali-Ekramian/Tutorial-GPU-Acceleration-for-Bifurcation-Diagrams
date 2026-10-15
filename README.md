# Tutorial: GPU Acceleration for Bifurcation Diagrams

A short, self-contained demo showing how `jax.vmap` + `jax.jit` dramatically speed up parameter scans. First by vectorizing (even on CPU), and then further by running on GPU. We use the **logistic map** bifurcation diagram as a simple, visual example, but the same pattern applies directly to other ODE-based bifurcation diagrams (like my oscillator network, γx/γy/W scans).

## The model

$$x_{n+1} = r \cdot x_n (1 - x_n)$$

We scan over 50,000 values of `r ∈ [2.8, 4.0]`, iterate the map 2,000 times per value, and plot the last 500 iterations of each trajectory to reveal the long-term (steady-state/chaotic) behavior.

## Three versions compared

| Version | Description | Runtime (this sample) |
|---|---|---|
| **1. CPU loop** | Plain Python `for` loop over every `r`, every iteration | ~ 47 s |
| **2. GPU code, run on CPU runtime** | Same `vmap` + `jit` code, but Colab runtime set to CPU | ~ 1 s |
| **3. GPU code, run on GPU runtime** | Same code, Colab runtime set to GPU (T4) | ~ 0.03 s |

**Total speedup CPU loop → GPU: ~1,570x**

## Files

- `cpu_version.ipynb` : naive nested loop (version 1)
- `gpu_version.ipynb` : JAX `vmap`+`jit` version (versions 2 & 3; only the Colab runtime setting differs)

## How to run

1. Open both scripts in Google Colab.
2. Run `cpu_version.py` as-is, no special setup needed.
3. For `gpu_version.py`:
   - **First run:** leave runtime as default (CPU) → this gives you result **#2**
   - **Then:** go to `Runtime → Change runtime type → Hardware accelerator → GPU (T4)`, restart, and re-run → this gives you result **#3**
4. Compare the three printed runtimes and the resulting plots (all three should look visually identical and only speed differs).

## Results

### Effect 1 — Vectorization & compilation (`vmap` + `jit`), independent of hardware

The naive loop does ~100 million individual Python scalar operations (50,000 r-values × 2,000 iterations), each paying the overhead of the Python interpreter (type checks, dynamic dispatch, function call overhead).

`jax.vmap` restructures this into **2,000 sequential steps, each operating on one array of 50,000 numbers at once**. `jax.jit` then compiles this into optimized machine code (via XLA), removing Python overhead entirely and enabling SIMD (processing several numbers per CPU instruction).

This is why version **2** (GPU code, run on CPU) is already ~47x faster than version **1**, before the GPU is even involved.

### Effect 2 — Hardware parallelism (GPU cores)

At each of the 2,000 steps, the array of 50,000 numbers still needs to be processed. A CPU can only process a handful of values at a time in parallel (limited by core count and SIMD width). A GPU (T4) has thousands of cores, so it can process (nearly) all 50,000 values in that step simultaneously.

This is why version **3** (GPU code, run on GPU) is another ~30x faster than version **2**. same code, just more parallel hardware underneath.

## Takeaway

> **Step 1 — Vectorize your code** (`vmap`/`jit`): this helps a lot, even without a GPU.
> 
> **Step 2 — Run it on GPU**: this helps *more*, and the benefit grows with the size of your parameter grid.
