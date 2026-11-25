import numpy as np
from surface_fairing.io_.artifacts import rippled_plane, rippled_curved
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.curvature import curvature_field

def locks(s):
    h = HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h

def stats(s):
    _,_,f = curvature_field(s,33,33)
    return (np.sqrt(np.mean(f['mean']**2)), np.abs(f['mean']).max(),
            np.sqrt(np.mean(f['gaussian']**2)), np.abs(f['gaussian']).max())

for name, art in (('plane',rippled_plane()),('curved',rippled_curved())):
    s = art.surface
    h0 = stats(s)
    print(f'== {name} ==  before H_rms={h0[0]:.4f} H_max={h0[1]:.4f} K_rms={h0[2]:.4f} K_max={h0[3]:.4f}')
    z0 = s.evaluate(.5,.5)
    for mode in ('jerk','normal','thin_plate'):
        r = fair_surface(s, locks(s), bend_mode=mode, attachment_weight=0.6,
                         memb_weight=0.05)
        h1 = stats(r.faired)
        z1 = r.faired.evaluate(.5,.5)
        print(f'  {mode:10s} E {r.direct_before:8.4f}->{r.direct_after:8.4f} '
              f'H_rms {h0[0]:.3f}->{h1[0]:.3f} H_max {h0[1]:.3f}->{h1[1]:.3f} '
              f'K_rms {h0[2]:.3f}->{h1[2]:.3f} disp={r.position_residual:.4f} '
              f'center z {z0[2]:.4f}->{z1[2]:.4f} hard_err={r.hard_constraint_error:.1e}')
