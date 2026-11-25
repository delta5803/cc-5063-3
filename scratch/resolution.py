import numpy as np
from surface_fairing.io_.artifacts import rippled_curved, _greville_grid
from surface_fairing.core.bspline import BSplineSurface
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface

def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
def make(n,waves,amp=0.03,curv=0.25):
    G=_greville_grid(n,n)
    base=curv*np.sin(np.pi*G[0])*np.sin(np.pi*G[1])
    rip=amp*np.sin(2*np.pi*waves*G[0])*np.cos(2*np.pi*waves*G[1])
    rip[0,:]=rip[-1,:]=rip[:,0]=rip[:,-1]=0
    dome=BSplineSurface(np.stack([G[0],G[1],base],axis=-1))
    s=BSplineSurface(np.stack([G[0],G[1],base+rip],axis=-1))
    return s,dome
def rms_to(surf,ref,grid=48):
    us=vs=np.linspace(0,1,grid)
    d=[np.linalg.norm(surf.evaluate(u,v)-ref.evaluate(u,v)) for u in us for v in vs]
    return np.sqrt(np.mean(np.square(d)))
for n,waves in ((10,4),(16,6),(20,8)):
    s,dome=make(n,waves)
    inp=rms_to(s,dome)
    print(f'grid {n} waves {waves}: input-to-dome rms {inp:.5f}')
    for att in (0.02,0.1,0.4):
        r=fair_surface(s,locks(s),bend_mode='jerk',attachment_weight=att)
        out=rms_to(r.faired,dome)
        print(f'   att {att:4.2f}  to-dome rms {out:.5f}  disp {r.position_residual:.5f}  ripple removal {1-out/inp:+.0%}')
