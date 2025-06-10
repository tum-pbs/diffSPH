from typing import List
import numpy as np
from typing import Union, Tuple, Optional, List
import torch
from dataclasses import dataclass
from sphMath.util import cloneOptionalTensor
def cloneOptionalTensor(T : Optional[torch.Tensor]):
    if T is None:
        return None
    else:
        return T.clone()
def detachOptionalTensor(T : Optional[torch.Tensor]):
    if T is None:
        return None
    else:
        return T.detach()

from sphMath.util import KernelTerms
from sphMath.schemes.states.common import BasicState

@torch.jit.script
@dataclass(slots=True)
class CompressibleState:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor    
    velocities: torch.Tensor
    
    # meta properties for particles
    kinds: torch.Tensor # 0 for fluid, 1 for boundary, 2 for ghost
    materials: torch.Tensor # specific subsets of particles, also indicates body id
    UIDs : torch.Tensor # unique identifiers for particles, ghost particles have negative UIDs

    # compressible state
    internalEnergies: Optional[torch.Tensor]
    totalEnergies: Optional[torch.Tensor]
    entropies: Optional[torch.Tensor]
    pressures: Optional[torch.Tensor]
    soundspeeds: Optional[torch.Tensor]
    
    UIDcounter: int = 0
    # cullen viscosity switch
    alphas: Optional[torch.Tensor] = None
    alpha0s: Optional[torch.Tensor] = None
    divergence : Optional[torch.Tensor] = None

    # PSPH specific terms
    n: Optional[torch.Tensor] = None
    dndh: Optional[torch.Tensor] = None
    P: Optional[torch.Tensor] = None
    dPdh: Optional[torch.Tensor] = None
    
    # CRK Kernel Correction Terms
    A: Optional[torch.Tensor] = None
    B: Optional[torch.Tensor] = None
    gradA: Optional[torch.Tensor] = None
    gradB: Optional[torch.Tensor] = None

    # Kernel Gradient Renormalization
    gradCorrectionMatrices: Optional[torch.Tensor] = None

    # grad-H Term:
    omega: Optional[torch.Tensor] = None

    # CRKSPH specific terms
    # A: Optional[torch.Tensor] = None
    # B: Optional[torch.Tensor] = None
    # gradA: Optional[torch.Tensor] = None
    # gradB: Optional[torch.Tensor] = None
    # V: Optional[torch.Tensor] = None
    apparentArea : Optional[torch.Tensor] = None

    # adaptive resolution corrector
    # omega: Optional[torch.Tensor] = None   
    # # useful for compatibility reasons for now
    # quantities: Optional[torch.Tensor] = None

    # Compatible terms
    ap_ij : Optional[torch.Tensor] = None
    av_ij : Optional[torch.Tensor] = None
    f_ij : Optional[torch.Tensor] = None

    # Ghost terms
    ghostIndices: Optional[torch.Tensor] = None
    ghostOffsets: Optional[torch.Tensor] = None

    # Neighborhood terms
    numNeighbors: Optional[torch.Tensor] = None

    # # Neighborhood
    # adjacency: Optional[SparseCOO] = None
    # species: Optional[torch.Tensor] = None

    # def _replace(self, **kwargs):
    #     return dataclasses.replace(self, **kwargs)
    
    def initializeNewState(self, scheme : str = 'internalEnergy'):
        u: Optional[torch.Tensor] = None
        E: Optional[torch.Tensor] = None
        s: Optional[torch.Tensor] = None
        p: Optional[torch.Tensor] = None
        c: Optional[torch.Tensor] = None
        if 'internalenergy' in scheme.lower():
            assert self.internalEnergies is not None, "Internal energy is not set"
            u = cloneOptionalTensor(self.internalEnergies)
        if 'totalenergy' in scheme.lower():
            assert self.totalEnergies is not None, "Total energy is not set"
            E = cloneOptionalTensor(self.totalEnergies)
        if 'entropy' in scheme.lower():
            assert self.entropies is not None, "Entropy is not set"
            s = cloneOptionalTensor(self.entropies)
        if 'pressure' in scheme.lower():
            assert self.pressures is not None, "Pressure is not set"
            p = cloneOptionalTensor(self.pressures)
        if 'soundspeed' in scheme.lower():
            assert self.soundspeeds is not None, "Soundspeed is not set"
            c = cloneOptionalTensor(self.soundspeeds)

        return CompressibleState(
            positions = self.positions.clone(),
            supports = self.supports.clone(),
            masses = self.masses.clone(),
            densities = self.densities.clone(),
            velocities = self.velocities.clone(),
            
            kinds = self.kinds.clone(),
            materials = self.materials.clone(),
            UIDs = self.UIDs.clone(),
            UIDcounter= self.UIDcounter,

            internalEnergies = u, #self.internalEnergies.clone() if 'internalenergy' in scheme.lower() else None,
            totalEnergies = E, #self.totalEnergies.clone() if 'totalenergy' in scheme.lower() else None,
            entropies = s, #self.entropies.clone() if 'entropy' in scheme.lower() else None,
            pressures = p, #self.pressures.clone() if 'pressure' in scheme.lower() else None,
            soundspeeds = c, #self.soundspeeds.clone() if 'soundspeed' in scheme.lower() else None,

            alphas = None,
            alpha0s= cloneOptionalTensor(self.alpha0s),
            divergence = cloneOptionalTensor(self.divergence),
            ghostIndices = cloneOptionalTensor(self.ghostIndices),
            ghostOffsets = cloneOptionalTensor(self.ghostOffsets),

            n = None,
            dndh = None,
            P = None,
            dPdh = None,

            # A = None,
            # B = None,
            # gradA = None,
            # gradB = None,
            # V = None,
            A = None,
            B = None,
            gradA = None,
            gradB = None,
            gradCorrectionMatrices = None,
            omega = None,
            apparentArea=None,
            # omega = None,
            ap_ij= None,
            av_ij= None,
            f_ij= None,
            numNeighbors=None
            
            # species = self.species.clone() if self.species is not None else None,
        )
    
    def nograd(self):
        return CompressibleState(
            positions = self.positions.detach().clone(),
            supports = self.supports.detach().clone(),
            masses = self.masses.detach().clone(),
            densities = self.densities.detach().clone(),
            velocities = self.velocities.detach().clone(),

            kinds = self.kinds.detach().clone(),
            materials = self.materials.detach().clone(),
            UIDs = self.UIDs.detach().clone(),
            UIDcounter= self.UIDcounter,

            internalEnergies = detachOptionalTensor(self.internalEnergies),
            totalEnergies = detachOptionalTensor(self.totalEnergies),
            entropies = detachOptionalTensor(self.entropies),
            pressures = detachOptionalTensor(self.pressures),
            soundspeeds = detachOptionalTensor(self.soundspeeds),
            alphas = detachOptionalTensor(self.alphas),
            alpha0s = detachOptionalTensor(self.alpha0s),
            divergence = detachOptionalTensor(self.divergence),
            ghostIndices = detachOptionalTensor(self.ghostIndices),
            ghostOffsets = detachOptionalTensor(self.ghostOffsets),
            n = detachOptionalTensor(self.n),
            dndh = detachOptionalTensor(self.dndh), 
            P = detachOptionalTensor(self.P),
            dPdh = detachOptionalTensor(self.dPdh),
            # A = detachOptionalTensor(self.A),
            # B = detachOptionalTensor(self.B),
            # gradA = detachOptionalTensor(self.gradA),
            # gradB = detachOptionalTensor(self.gradB),
            apparentArea= detachOptionalTensor(self.apparentArea),
            # omega = detachOptionalTensor(self.omega),
            A=detachOptionalTensor(self.A),
            B=detachOptionalTensor(self.B),
            gradA=detachOptionalTensor(self.gradA),
            gradB=detachOptionalTensor(self.gradB),
            gradCorrectionMatrices=detachOptionalTensor(self.gradCorrectionMatrices),
            omega=detachOptionalTensor(self.omega),
            ap_ij = detachOptionalTensor(self.ap_ij),
            av_ij = detachOptionalTensor(self.av_ij),
            f_ij = detachOptionalTensor(self.f_ij),
            numNeighbors = detachOptionalTensor(self.numNeighbors),
        )
    def clone(self):
        return CompressibleState(
            positions = self.positions.clone(),
            supports = self.supports.clone(),
            masses = self.masses.clone(),
            densities = self.densities.clone(),
            velocities = self.velocities.clone(),

            kinds = self.kinds.clone(),
            materials = self.materials.clone(),
            UIDs = self.UIDs.clone(),
            UIDcounter= self.UIDcounter,

            internalEnergies = cloneOptionalTensor(self.internalEnergies),
            totalEnergies = cloneOptionalTensor(self.totalEnergies),
            entropies = cloneOptionalTensor(self.entropies),
            pressures = cloneOptionalTensor(self.pressures),
            soundspeeds = cloneOptionalTensor(self.soundspeeds),
            alphas = cloneOptionalTensor(self.alphas),
            alpha0s = cloneOptionalTensor(self.alpha0s),
            divergence = cloneOptionalTensor(self.divergence),
            ghostIndices = cloneOptionalTensor(self.ghostIndices),
            ghostOffsets = cloneOptionalTensor(self.ghostOffsets),
            n = cloneOptionalTensor(self.n),
            dndh = cloneOptionalTensor(self.dndh), 
            P = cloneOptionalTensor(self.P),
            dPdh = cloneOptionalTensor(self.dPdh),
            # A = detachOptionalTensor(self.A),
            # B = detachOptionalTensor(self.B),
            # gradA = detachOptionalTensor(self.gradA),
            # gradB = detachOptionalTensor(self.gradB),
            apparentArea= cloneOptionalTensor(self.apparentArea),
            # omega = detachOptionalTensor(self.omega),
            A=cloneOptionalTensor(self.A),
            B=cloneOptionalTensor(self.B),
            gradA=cloneOptionalTensor(self.gradA),
            gradB=cloneOptionalTensor(self.gradB),
            gradCorrectionMatrices=cloneOptionalTensor(self.gradCorrectionMatrices),
            omega=cloneOptionalTensor(self.omega),
            ap_ij = cloneOptionalTensor(self.ap_ij),
            av_ij = cloneOptionalTensor(self.av_ij),
            f_ij = cloneOptionalTensor(self.f_ij),
            numNeighbors = cloneOptionalTensor(self.numNeighbors),
        )

@dataclass(slots=True)
class CompressibleUpdate:
    positions: torch.Tensor
    velocities: torch.Tensor
    totalEnergies: Optional[torch.Tensor] = None
    internalEnergies: Optional[torch.Tensor] = None
    densities: Optional[torch.Tensor] = None
    passive: Optional[torch.Tensor] = None
