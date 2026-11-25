import numpy as np
from surface_fairing.io_.artifacts import rippled_curved, _greville_grid
from surface_fairing.core.bspline import BSplineSurface
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.curvature import curvature_field

def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h

nu=nv=10
G=_greville_grid(nu,nv)
curv=0.25
base_z=curv*np.sin(np.pi*G[0])*np.sin(np.pi*G[1])
base=BSplineSurface(np.stack([G[0],G[1],base_z],axis=-1),name='base_dome')

art=rippled_curved(); s=art.surface
grid=33
us=vs=np.linspace(0,1,grid)
def rms_to(surf,ref):
    d=[np.linalg.norm(surf.evaluate(u,v)-ref.evaluate(u,v)) for u in us for v in vs]
    return np.sqrt(np.mean(np.square(d))), np.max(d)
_,_,fc=curvature_field(s,grid,grid)
_,_,fb=curvature_field(base,grid,grid)
print('rippled vs base: rms %.5f max %.5f | H_rms rippled %.4f base %.4f' % (
    *rms_to(s,base), np.sqrt(np.mean(fc['mean']**2)), np.sqrt(np.mean(fb['mean']**2))))
for mode in ('jerk','normal','thin_plate'):
    r=fair_surface(s,locks(s),bend_mode=mode,attachment_weight=0.6,memb_weight=0.05)
    rms,mx=rms_to(r.faired,base)
    _,_,fa=curvature_field(r.faired,grid,grid)
    print('%-10s to-base rms %.5f max %.5f | H_rms %.4f (base %.4f) K_rms %.4f (base %.4f) | z_center %.4f base %.4f' % (
        mode,rms,mx,np.sqrt(np.mean(fa['mean']**2)),np.sqrt(np.mean(fb['mean']**2)),
        np.sqrt(np.mean(fa['gaussian']**2)),np.sqrt(np.mean(fb['gaussian']**2)),
        r.faired.evaluate(.5,.5)[2], base.evaluate(.5,.5)[2]))
