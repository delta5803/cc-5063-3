import numpy as np
from surface_fairing.core.bspline import BSplineSurface, greville_abscissae
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.energy import assemble_energy
from surface_fairing.core.solver import solve_qp
import sys; sys.path.insert(0,'scripts')
from measure import _high_frequency_rms
def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h
def plane(n,waves,amp=0.04):
    g=greville_abscissae(n,3); U,V=np.meshgrid(g,g,indexing='ij')
    rip=amp*np.sin(2*np.pi*waves*U)*np.sin(2*np.pi*waves*V)
    rip[0,:]=rip[-1,:]=rip[:,0]=rip[:,-1]=0
    return BSplineSurface(np.stack([U,V,rip],axis=-1))

# Custom solve: minimize 1/2 x'Hx + att/2 ||x-xref||^2  s.t. Cx=b
# Build a pseudo EnergyModel with H (already normalized), then call a manual
# null-space solve that uses the full linear term H x (not zero).
def fair_absolute(s, mode, att):
    m=assemble_energy(s,bend_mode=mode); H=m.H; xref=m.x_ref
    C,b=locks(s).matrix()
    U,S,Vh=np.linalg.svd(C,full_matrices=True)
    r=int(np.sum(S>1e-10*S[0])); Z=Vh[r:].T
    sinv=np.where(S[:r]>0,1/S[:r],0.0)
    z0=Vh[:r].T@(sinv*(U[:,:r].T@b))
    Ht=H+att*np.eye(H.shape[0])
    # x=z0+Zy; gradient: Ht x - att xref =0
    rH=Z.T@Ht@Z
    rg=Z.T@(att*xref-Ht@z0)
    y=np.linalg.solve(rH,rg)
    x=z0+Z@y
    out=s.clone(); out.set_flat(x); return out

for n,waves in ((16,3),(16,5),(20,4)):
    s=plane(n,waves); hf0=_high_frequency_rms(s)
    for att in (0.02,0.2):
        f=fair_absolute(s,'jerk',att)
        print('ABS grid %2d wave %2d att %.2f HF %.5f->%.5f (%4.1f%%) disp %.5f' % (
            n,waves,att,hf0,_high_frequency_rms(f),100*(1-_high_frequency_rms(f)/hf0),
            np.max(np.linalg.norm(f.P-s.P,axis=-1))))
