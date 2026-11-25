import numpy as np
from surface_fairing.io_.artifacts import rippled_curved, rippled_plane
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.energy import assemble_energy
from surface_fairing.core.solver import solve_qp
from surface_fairing.core.bspline import BSplineSurface
import sys; sys.path.insert(0,'scripts')
from measure import _high_frequency_rms,_smooth_dome,_rms_distance
def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
def blended(s,wj,wn,att):
    mj=assemble_energy(s,bend_mode='jerk'); mn=assemble_energy(s,bend_mode='normal')
    # both normalized to order-one; combine dimensionless matrices
    H=wj*mj.H+wn*mn.H
    # reuse solver via a stand-in model object
    class M: pass
    m=M(); m.H=H; m.x_ref=mj.x_ref
    rep=solve_qp(H,mj.x_ref,locks(s),attachment_weight=att)
    out=s.clone(); out.set_flat(rep.x); return out
for name,art in (('curved',rippled_curved()),('plane',rippled_plane())):
    s=art.surface; hf0=_high_frequency_rms(s)
    base=_smooth_dome(art) if name=='curved' else None
    print('==',name,'HF0 %.5f'%hf0)
    for wj,wn in ((1.0,0.0),(0.75,0.25),(0.5,0.5),(0.25,0.75),(0.0,1.0)):
        f=blended(s,wj,wn,0.05)
        extra=''
        if base is not None:
            extra=' rms-base %.5f z %.4f' % (_rms_distance(f,base),f.evaluate(.5,.5)[2])
        print('  wj %.2f wn %.2f HF %.5f (%4.1f%%) disp %.4f%s' % (
            wj,wn,_high_frequency_rms(f),100*(1-_high_frequency_rms(f)/hf0),
            np.max(np.linalg.norm(f.P-s.P,axis=-1)),extra))
