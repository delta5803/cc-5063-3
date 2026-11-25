import numpy as np
from surface_fairing.io_.artifacts import rippled_plane
from surface_fairing.core.constraints import HardConstraints, make_ring_soft_lock
from surface_fairing.core.fairing import fair_surface
import sys; sys.path.insert(0,'scripts')
from measure import _high_frequency_rms
s=rippled_plane().surface
h=HardConstraints(3*s.n_control)
for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
for soft in ([], [make_ring_soft_lock(s,'u0',100.0)]):
    r=fair_surface(s,h,soft=soft,bend_mode='normal',attachment_weight=0.3)
    print('soft=%s HF %.6f disp %.5f events=%d' % (
        bool(soft),_high_frequency_rms(r.faired),r.position_residual,len(r.soft_events)))
