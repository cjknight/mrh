from gpu4mrh.df.df_jk import _get_jk, evict_dfobj_later
from pyscf.df import df_jk

from gpu4mrh.lib.utils import patch_cpu_kernel

print(f'{df_jk} monkey-patched to GPU accelerated version')
df_jk.get_jk = patch_cpu_kernel(df_jk.get_jk)(_get_jk)

from pyscf.df.df import DF
from gpu4mrh.lib.utils import resolve_use_gpu
from mrh.my_pyscf.gpu import libgpu

_cpu_reset = DF.reset

def _reset(self, mol=None):
    '''DF.reset is where PySCF drops _cderi when the geometry changes (scanner
    mode: geometry optimizers, scans, finite differences). The device ERI cache
    is keyed on id(dfobj)+block count, and the DF object survives the reset, so
    flag the cache for refresh on the next get_jk. The flag is cleared in
    pull_get_jk.'''
    out = _cpu_reset(self, mol)
    gpu = resolve_use_gpu(self)
    if gpu is not None:
        libgpu.set_update_dfobj_(gpu, 2) # 2: also refresh full-molecule dfobj
        evict_dfobj_later(gpu, id(self)) # blocks may change shape; free them on the next get_jk
    return out

DF.reset = _reset
print(f'{DF}.reset monkey-patched to invalidate GPU ERI cache')
