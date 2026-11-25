import numpy as np
from surface_fairing.core.bspline import BSplineSurface, greville_abscissae
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
def make_hard(s, rings):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'):
        h.pin_edge(s,e)
        if rings: h.pin_ring(s,e)
    return h
def plane(n,waves,amp=0.04):
    g=greville_abscissae(n,3); U,V=np.meshgrid(g,g,indexing='ij')
    rip=amp*np.sin(2*np.pi*waves*U)*np.sin(2*np.pi*waves*V)
    rip[0,:]=rip[-1,:]=rip[:,0]=rip[:,-1]=0
    return BSplineSurface(np.stack([U,V,rip],axis=-1))
def zrms(s,grid=64):
    ts=np.linspace(0,1,grid)
    z=np.array([s.evaluate(u,v)[2] for u in ts for v in ts])
    return np.sqrt(np.mean(z**2))
for n,waves in ((16,3),(20,4)):
    s=plane(n,waves); z0=zrms(s)
    for rings in (False,True):
        r=fair_surface(s,make_hard(s,rings),bend_mode='jerk',attachment_weight=0.02)
        print('grid%2d wave%2d ringlock=%s z_rms %.5f->%.5f (%4.1f%% removed)' % (
            n,waves,rings,z0,zrms(r.faired),100*(1-zrms(r.faired)/z0)))
