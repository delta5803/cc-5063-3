import numpy as np
from surface_fairing.io_.artifacts import rippled_curved, rippled_plane, _greville_grid
from surface_fairing.core.bspline import BSplineSurface
from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface

def locks(s):
    h=HardConstraints(3*s.n_control)
    for e in ('u0','u1','v0','v1'): h.pin_edge(s,e)
    return h

# high-frequency content = deviation from a coarse (smoothed) version via
# spectral cutoff on the sampled z-field; report the HF rms before/after.
def hf_rms(surf, grid=64, cutoff=3):
    us=vs=np.linspace(0,1,grid)
    Z=np.array([[surf.evaluate(u,v)[2] for v in vs] for u in us])
    F=np.fft.fft2(Z-Z.mean())
    K=np.minimum(np.arange(grid),grid-np.arange(grid))
    KU,KV=np.meshgrid(K,K,indexing='ij')
    mask=(np.abs(KU)>=cutoff)|(np.abs(KV)>=cutoff)
    return np.sqrt(np.mean((np.real(np.fft.ifft2(F*mask)))**2))

for name,art,att in (('plane',rippled_plane(),0.3),('curved',rippled_curved(),0.2)):
    s=art.surface
    print('==',name,' HF rms before %.6f'%hf_rms(s))
    for mode in ('jerk','normal','thin_plate'):
        r=fair_surface(s,locks(s),bend_mode=mode,attachment_weight=att)
        print('   %-10s HF rms after %.6f  disp %.5f' % (
            mode,hf_rms(r.faired),r.position_residual))
