# Tutorial-GPU-Acceleration-for-Bifurcation-Diagrams
A short, self-contained demo showing how `jax.vmap` + `jax.jit` dramatically speed up parameter scans. First by vectorizing (even on CPU), and then further by running on GPU. We use the **logistic map** bifurcation diagram as a simple, visual example, but the same pattern applies directly to other ODE-based bifurcation diagrams.
