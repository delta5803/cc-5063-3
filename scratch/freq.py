import numpy as np
from surface_fairing.core.bspline import BSplineSurface, greville_abscissae
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
import sys; sys.path.insert(0,'scripts')
from measure import _high_frequency_rms
def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
def build(n,waves,amp=0.03):
    g=greville_abscissae(n,3); U,V=np.meshgrid(g,g,indexing='ij')
    base=0.25*np.sin(np.pi*U)*np.sin(np.pi*V)
    rip=amp*np.sin(2*np.pi*waves*U)*np.cos(2*np.pi*waves*V)
    rip[0,:]=rip[-1,:]=rip[:,0]=rip[:,-1]=0
    return BSplineSurface(np.stack([U,V,base+rip],axis=-1)), BSplineSurface(np.stack([U,V,base],axis=-1))
import sys; sys.path.insert(0,'scripts')
from measure import _rms_distance
for n,waves in ((20,4),(20,6),(20,8),(24,8),(24,10)):
    s,dome=build(n,waves); hf0=_high_frequency_rms(s)
    r=fair_surface(s,locks(s),bend_mode='jerk',attachment_weight=0.05)
    hf=_high_frequency_rms(r.faired)
    print('grid %2d waves %2d | HF %.5f->%.5f (%4.1f%% removed) rms-base %.5f->%.5f disp %.4f' % (
        n,waves,hf0,hf,100*(1-hf/hf0),_rms_distance(s,dome),_rms_distance(r.faired,dome),r.position_residual))
