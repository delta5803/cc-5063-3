import numpy as np
from surface_fairing.io_.artifacts import rippled_curved
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.curvature import curvature_field

def locks(s):
    h = HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h

art = rippled_curved(); s = art.surface
_,_,fb = curvature_field(s,33,33)
z0 = s.evaluate(.5,.5)[2]
print('before H_rms %.4f K_rms %.4f center_z %.4f' % (
    np.sqrt(np.mean(fb['mean']**2)), np.sqrt(np.mean(fb['gaussian']**2)), z0))
for att in (0.05,0.2,0.6,2.0,6.0,20.0):
  for memb in (0.0,0.05,0.3):
    r = fair_surface(s, locks(s), bend_mode='jerk', attachment_weight=att,
                     memb_weight=memb)
    _,_,fa = curvature_field(r.faired,33,33)
    z1 = r.faired.evaluate(.5,.5)[2]
    print('att %5.2f memb %4.2f | H_rms %.3f K_rms %.3f Hmax %.3f | disp %.4f z %.4f->%.4f (%.1f%%)' % (
        att,memb,np.sqrt(np.mean(fa['mean']**2)),np.sqrt(np.mean(fa['gaussian']**2)),
        np.abs(fa['mean']).max(),r.position_residual,z0,z1,100*(z0-z1)/z0))
