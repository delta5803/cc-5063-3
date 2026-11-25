import numpy as np
from surface_fairing.io_.artifacts import over_fairing_case
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
def locks(s,key=False):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    if key: h.pin_surface_point(s,.5,.5)
    return h
art=over_fairing_case(); s=art.surface
kh=art.expected['key_height']
print('key height',kh)
for att in (0.0,0.001,0.005,0.02,0.05,0.2):
    r=fair_surface(s,locks(s),bend_mode='jerk',attachment_weight=max(att,1e-9))
    print('no-lock att %.4f key z %.5f disp %.5f' % (att,r.faired.evaluate(.5,.5)[2],r.position_residual))
rk=fair_surface(s,locks(s,True),bend_mode='jerk',attachment_weight=0.3)
print('locked key z %.6f (target %.6f)' % (rk.faired.evaluate(.5,.5)[2],kh))
