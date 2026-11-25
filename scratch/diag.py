import numpy as np
from surface_fairing.core.bspline import BSplineSurface, greville_abscissae
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.energy import assemble_energy
from surface_fairing.core.solver import _constraint_decomposition
import sys; sys.path.insert(0,'scripts')
from measure import _high_frequency_rms
def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
# pure ripple on a PLANE, no dome, varying frequency
def plane(n,waves,amp=0.04):
    g=greville_abscissae(n,3); U,V=np.meshgrid(g,g,indexing='ij')
    rip=amp*np.sin(2*np.pi*waves*U)*np.sin(2*np.pi*waves*V)
    rip[0,:]=rip[-1,:]=rip[:,0]=rip[:,-1]=0
    return BSplineSurface(np.stack([U,V,rip],axis=-1))
for n,waves in ((12,2),(12,3),(16,3),(16,5),(20,4)):
    s=plane(n,waves); hf0=_high_frequency_rms(s)
    # eigen stiffness in free null space
    model=assemble_energy(s,bend_mode='jerk')
    C,b=locks(s).matrix()
    z0,Z=_constraint_decomposition(C,b,1e-10,3*s.n_control)
    rH=Z.T@model.H@Z
    eig=np.linalg.eigvalsh(rH)
    r=fair_surface(s,locks(s),bend_mode='jerk',attachment_weight=0.02)
    hf=_high_frequency_rms(r.faired)
    print('grid %2d waves %2d freeDOF %3d eig[min,med,max] %.3e %.3e %.3e | HF %.5f->%.5f (%4.1f%%)' % (
        n,waves,Z.shape[1],eig[0],np.median(eig),eig[-1],hf0,hf,100*(1-hf/max(hf0,1e-12))))
