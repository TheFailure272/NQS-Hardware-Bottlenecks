# THE OMEGA ENGINE: Var-NiTE (Variational Neural Imaginary Time Evolution)
import os, jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

def build_pde_solver(chi, get_neural_tensor, damping=1e-3):
    @jax.jit
    def project_step(params, gate):
        flat_params, unflatten = ravel_pytree(params)
        # Matrix-Free Conjugate Gradient solving (J^T J + damping * I) * d_params = J^T * dA
        # (See Kaggle notebook for full implementation)
        return new_params
    return project_step

if __name__ == "__main__":
    print("[Engine] Booting Var-NiTE PDE Solver...")
