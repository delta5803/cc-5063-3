import numpy as np
from surface_fairing.io_.artifacts import rippled_curved, rippled_plane
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.constraints import HardConstraints
import sys; sys.path.insert(0,'scripts')
from measure import _high_frequency_rms,_smooth_dome,_rms_distance,_curvature_stats
def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
# curved: compare jerk (shape-safe) vs normal at a conservative attachment
art=rippled_curved(); s=art.surface; base=_smooth_dome(art)
print('CURVED input: rms-base %.5f HF %.5f H_rms %.4f' % (
    _rms_distance(s,base),_high_frequency_rms(s),_curvature_stats(s)['mean_rms']))
for mode in ('jerk','normal'):
  for att in (0.1,0.3,0.6):
    r=fair_surface(s,locks(s),bend_mode=mode,attachment_weight=att)
    print('  %-6s att %.1f rms-base %.5f HF %.5f (%4.1f%%) H_rms %.4f z %.4f disp %.4f' % (
        mode,att,_rms_distance(r.faired,base),_high_frequency_rms(r.faired),
        100*(1-_high_frequency_rms(r.faired)/_high_frequency_rms(s)),
        _curvature_stats(r.faired)['mean_rms'],r.faired.evaluate(.5,.5)[2],r.position_residual))
