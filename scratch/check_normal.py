import numpy as np
from surface_fairing.io_.artifacts import rippled_plane, rippled_curved
from surface_fairing.core.constraints import HardConstraints, make_ring_soft_lock
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.curvature import curvature_field
for art, att, mem, soft in ((rippled_plane(), 0.4, 0.0, False),
                            (rippled_curved(), 0.6, 0.05, True)):
    s = art.surface
    h = HardConstraints(3 * s.n_control)
    for e in ('u0', 'u1', 'v0', 'v1'):
        h.pin_edge(s, e)
    sl = [make_ring_soft_lock(s, 'u0', 100.0)] if soft else []
    r = fair_surface(s, h, soft=sl, attachment_weight=att,
                     memb_weight=mem, bend_mode='normal')
    _, _, b = curvature_field(s)
    _, _, a = curvature_field(r.faired)
    print(art.kind,
          'normal E %.4f->%.4f' % (r.direct_before, r.direct_after),
          'H rms %.3f->%.3f' % (np.sqrt((b['mean']**2).mean()),
                                np.sqrt((a['mean']**2).mean())),
          'Hmax %.3f->%.3f' % (np.abs(b['mean']).max(),
                               np.abs(a['mean']).max()),
          'Krms %.3f->%.3f' % (np.sqrt((b['gaussian']**2).mean()),
                               np.sqrt((a['gaussian']**2).mean())))
