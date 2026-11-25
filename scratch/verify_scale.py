import numpy as np
from surface_fairing.io_.artifacts import rippled_curved
from surface_fairing.core.energy import assemble_energy, direct_energy
for mode in ('jerk','normal','thin_plate'):
    s=rippled_curved().surface
    m=assemble_energy(s,bend_mode=mode)
    x=s.flatten().reshape(-1)
    quad=m.absolute_energy(x)            # already multiplies energy_scale back
    direct=direct_energy(s,bend_mode=mode)
    print('%-10s scale %.4e quad %.6f direct %.6f rel %.2e' % (
        mode,m.energy_scale,quad,direct,abs(quad-direct)/direct))
