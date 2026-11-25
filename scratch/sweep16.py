import numpy as np
from surface_fairing.io_.artifacts import rippled_curved
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.curvature import curvature_field
import sys; sys.path.insert(0,'scripts')
from measure import _smooth_dome,_rms_distance,_high_frequency_rms
def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
art=rippled_curved(); s=art.surface; base=_smooth_dome(art)
hf0=_high_frequency_rms(s); rms0=_rms_distance(s,base)
print('input HF %.5f rms-to-base %.5f' % (hf0,rms0))
for mode in ('jerk','thin_plate'):
  print('--',mode)
  for att in (0.005,0.01,0.02,0.05,0.1):
    r=fair_surface(s,locks(s),bend_mode=mode,attachment_weight=att)
    hf=_high_frequency_rms(r.faired); rms=_rms_distance(r.faired,base)
    _,_,fa=curvature_field(r.faired,33,33)
    print('  att %.3f HF %.5f (%.0f%% removed) rms-base %.5f disp %.4f H_rms %.3f z %.4f' % (
        att,hf,100*(1-hf/hf0),rms,r.position_residual,np.sqrt(np.mean(fa['mean']**2)),
        r.faired.evaluate(.5,.5)[2]))
