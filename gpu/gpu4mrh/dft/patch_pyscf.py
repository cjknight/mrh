from pyscf.dft import numint
from gpu4mrh.dft import numint as gpu_numint

# NumInt.nr_rks is the RKS XC integration used by dft.rks.get_veff. The GPU version resolves the
# device from mol and falls back to the CPU function for anything it does not handle.
numint.NumInt.nr_rks = gpu_numint.nr_rks
print(f'{numint.NumInt}.nr_rks monkey-patched to GPU accelerated version')
