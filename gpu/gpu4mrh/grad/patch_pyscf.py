from pyscf.grad import rks as rks_grad
from gpu4mrh.grad import rks as gpu_rks_grad

# grad.rks.get_veff and df.grad.rks.get_veff both look get_vxc up on the pyscf.grad.rks module
# at call time. The GPU version resolves the device from mol and falls back to the CPU function.
rks_grad.get_vxc = gpu_rks_grad.get_vxc
print(f'{rks_grad.__name__}.get_vxc monkey-patched to GPU accelerated version')
