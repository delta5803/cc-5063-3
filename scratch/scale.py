import numpy as np
from surface_fairing.io_.artifacts import rippled_curved
from surface_fairing.core.energy import assemble_energy
for mode in ('jerk','normal','thin_plate'):
    s=rippled_curved().surface
    m=assemble_energy(s,bend_mode=mode)
    d=np.diag(m.H)
    print('%-10s diag mean %.4e  min %.2e max %.2e  trace %.4e' % (
        mode,d.mean(),d.min(),d.max(),np.trace(m.H)))
