from gpu4mrh.df.df_jk import _get_jk
from pyscf.df import df_jk

from gpu4mrh.lib.utils import patch_cpu_kernel

print(f'{df_jk} monkey-patched to GPU accelerated version')
df_jk.get_jk = patch_cpu_kernel(df_jk.get_jk)(_get_jk)

from pyscf.df.df import DF
from mrh.my_pyscf.gpu import eri_cache

_cpu_reset = DF.reset

def _reset(self, mol=None):
    '''DF.reset is where PySCF drops _cderi when the geometry changes (scanner
    mode: geometry optimizers, scans, finite differences). The DF object survives
    the reset, so drop its blocks from the device ERI cache (keyed on id(dfobj)).'''
    eri_cache.evict(self)
    return _cpu_reset(self, mol)

DF.reset = _reset
print(f'{DF}.reset monkey-patched to evict the GPU ERI cache')

from gpu4mrh.lib.utils import resolve_use_gpu
from gpu4mrh.df import df_build

_cpu_build = DF.build

def _build(self):
    '''In-memory DF tensor build with the metric solve on the device (gpu4mrh.df.df_build); the
    result is pyscf's ordinary host _cderi. Builds pyscf writes to disk stay pyscf's.'''
    gpu = resolve_use_gpu(self)
    if gpu is None:
        return _cpu_build(self)
    return df_build.build(self, gpu, _cpu_build)

DF.build = _build
print(f'{DF}.build monkey-patched to GPU accelerated version (in-memory tensor)')

from pyscf.df.grad import rhf as df_rhf_grad
from gpu4mrh.df import grad_rhf as gpu_df_rhf_grad

# Gradients.get_jk (and through it get_k, the DF-RKS gradient's get_veff) looks get_jk up on the
# pyscf.df.grad.rhf module at call time. Falls back to the CPU function for what it does not handle.
df_rhf_grad.get_jk = gpu_df_rhf_grad.get_jk
print(f'{df_rhf_grad.__name__}.get_jk monkey-patched to GPU accelerated version')
