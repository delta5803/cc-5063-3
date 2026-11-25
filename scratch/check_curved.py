import numpy as np
from surface_fairing.io_.artifacts import rippled_curved
from surface_fairing.core.constraints import HardConstraints, make_ring_soft_lock
from surface_fairing.core.fairing import fair_surface
art = rippled_curved(); s = art.surface
for att in (0.6, 2.0, 6.0):
    h = HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    r = fair_surface(s, h, soft=[make_ring_soft_lock(s,'u0',100.0)],
                     attachment_weight=att, memb_weight=0.05, bend_mode='normal')
    f = r.faired
    print('att', att, 'center z %.4f->%.4f' % (s.evaluate(.5,.5)[2], f.evaluate(.5,.5)[2]),
          'disp %.4f' % r.position_residual,
          'E %.4f->%.4f' % (r.direct_before, r.direct_after))
# check reference normal vs faired normal deviation
h = HardConstraints(3*s.n_control)
for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
r = fair_surface(s, h, attachment_weight=2.0, memb_weight=0.05, bend_mode='normal')
nd = [np.arccos(np.clip(np.dot(s.normal(u,v), r.faired.normal(u,v)),-1,1))
      for u in np.linspace(.1,.9,9) for v in np.linspace(.1,.9,9)]
print('normal rotation max deg', np.degrees(max(nd)))
