import numpy as np
from surface_fairing.core.bspline import BSplineSurface, greville_abscissae
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
def plane(n,waves,amp=0.04):
    g=greville_abscissae(n,3); U,V=np.meshgrid(g,g,indexing='ij')
    rip=amp*np.sin(2*np.pi*waves*U)*np.sin(2*np.pi*waves*V)
    rip[0,:]=rip[-1,:]=rip[:,0]=rip[:,-1]=0
    flat=BSplineSurface(np.stack([U,V,np.zeros((n,n))],axis=-1))
    return BSplineSurface(np.stack([U,V,rip],axis=-1)),flat
def zrms(s,grid=64):
    ts=np.linspace(0,1,grid)
    z=np.array([s.evaluate(u,v)[2] for u in ts for v in ts])
    return np.sqrt(np.mean(z**2)),np.max(np.abs(z))
for n,waves in ((16,3),(20,4)):
    s,flat=plane(n,waves)
    z0,m0=zrms(s)
    print(f'grid{n} wave{waves} input z_rms {z0:.5f} max {m0:.5f}')
    for att in (0.001,0.01,0.05,0.5):
        r=fair_surface(s,locks(s),bend_mode='jerk',attachment_weight=att)
        z,m=zrms(r.faired)
        print(f'   att {att:.3f} z_rms {z:.5f} ({100*(1-z/z0):4.1f}% removed) max {m:.5f}')
