import numpy as np
from surface_fairing.io_.artifacts import rippled_curved, rippled_plane, _greville_grid
from surface_fairing.core.bspline import BSplineSurface
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.curvature import curvature_field

def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
nu=nv=10; G=_greville_grid(nu,nv)
base=BSplineSurface(np.stack([G[0],G[1],0.25*np.sin(np.pi*G[0])*np.sin(np.pi*G[1])],axis=-1))
grid=33; us=vs=np.linspace(0,1,grid)
def rms_to(surf,ref):
    d=[np.linalg.norm(surf.evaluate(u,v)-ref.evaluate(u,v)) for u in us for v in vs]
    return np.sqrt(np.mean(np.square(d))),np.max(d)
_,_,fb=curvature_field(base,grid,grid)
art=rippled_curved(); s=art.surface
print('rippled-to-base rms %.5f  base H_rms %.4f K_rms %.4f' % (
    rms_to(s,base)[0],np.sqrt(np.mean(fb['mean']**2)),np.sqrt(np.mean(fb['gaussian']**2))))
for att in (0.01,0.05,0.2,0.6,2.0,6.0):
    r=fair_surface(s,locks(s),bend_mode='jerk',attachment_weight=att)
    rms,mx=rms_to(r.faired,base); _,_,fa=curvature_field(r.faired,grid,grid)
    print('att %5.2f | to-base rms %.5f max %.5f | H_rms %.4f K_rms %.4f | z %.4f(base %.4f) disp %.4f' % (
        att,rms,mx,np.sqrt(np.mean(fa['mean']**2)),np.sqrt(np.mean(fa['gaussian']**2)),
        r.faired.evaluate(.5,.5)[2],base.evaluate(.5,.5)[2],r.position_residual))
# plane check with default-ish attachment
print('--- plane ---')
sp=rippled_plane().surface
for att in (0.05,0.3,1.0):
    r=fair_surface(sp,locks(sp),bend_mode='jerk',attachment_weight=att)
    _,_,fa=curvature_field(r.faired,grid,grid); _,_,f0=curvature_field(sp,grid,grid)
    print('att %.2f H_rms %.4f->%.4f K_rms %.4f->%.4f disp %.5f' % (
        att,np.sqrt(np.mean(f0['mean']**2)),np.sqrt(np.mean(fa['mean']**2)),
        np.sqrt(np.mean(f0['gaussian']**2)),np.sqrt(np.mean(fa['gaussian']**2)),r.position_residual))
