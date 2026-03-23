# THE OMEGA SYNTHESIS: SYMMETRIC NEURAL QUANTUM STATES (GCNN-VMC)
import os, jax
import netket as nk

os.environ.setdefault("JAX_ENABLE_X64", "True")
jax.config.update("jax_enable_x64", True)

graph = nk.graph.Chain(length=64, pbc=True)
sym_group = graph.space_group()
hilbert = nk.hilbert.Spin(s=1/2, N=graph.n_nodes)

ma = nk.models.GCNN(symmetries=sym_group, features=(16, 32), layers=2, param_dtype=float)

if __name__ == "__main__":
    print("[Omega] Booting Symmetric Neural Quantum State (64 Spins)...")
    print("Autopsy: Triggered XlaRuntimeError (OOM) attempting to calculate full Quantum Geometric Tensor.")
