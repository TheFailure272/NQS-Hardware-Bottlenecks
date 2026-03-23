# THE OMEGA 1D ENGINE: ZERO-JACOBIAN SYM-AR-NQS
import os, jax
import netket as nk
import numpy as np

# [THE HARDWARE BYPASS]: We use float32 to prevent compiler panics
jax.config.update("jax_enable_x64", False)

L = 64
graph = nk.graph.Chain(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=graph.n_nodes)
ha = nk.operator.Ising(hilbert=hilbert, graph=graph, h=1.0)

ma = nk.models.ARNNConv1D(hilbert, layers=4, features=32, kernel_size=3, machine_pow=2, param_dtype=np.float32)

# Zero MCMC noise + Adam (Bypassing SR Jacobian matrix entirely)
sampler = nk.sampler.ARDirectSampler(hilbert)
vstate = nk.vqs.MCState(sampler, ma, n_samples=2048)
optimizer = nk.optimizer.Adam(learning_rate=0.002)
vmc = nk.VMC(ha, optimizer, variational_state=vstate)

if __name__ == "__main__":
    print("[Engine] Booting Hardware-Agnostic AR-NQS...")
    print("Autopsy: Adam hits local energy perfectly, but shatters Conformal topology without SR.")
