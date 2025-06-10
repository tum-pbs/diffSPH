from typing import List
import numpy as np
from typing import Union, Tuple, Optional, List
import torch
from dataclasses import dataclass
from sphMath.util import cloneOptionalTensor
from sphMath.util import KernelTerms
def detachOptionalTensor(T : Optional[torch.Tensor]):
    if T is None:
        return None
    else:
        return T.detach()
@torch.jit.script
@dataclass(slots=True)
class WeaklyCompressibleState:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor    
    velocities: torch.Tensor
    
    pressures: torch.Tensor
    soundspeeds: torch.Tensor
    
    # meta properties for particles
    kinds: torch.Tensor # 0 for fluid, 1 for boundary, 2 for ghost
    materials: torch.Tensor # specific subsets of particles, also indicates body id
    UIDs : torch.Tensor # unique identifiers for particles, ghost particles have negative UIDs
    
    UIDcounter: int = 0

    # Delta-SPH terms
    covarianceMatrices: Optional[torch.Tensor] = None
    # gradCorrectionMatrices: Optional[torch.Tensor] = None
    eigenValues: Optional[torch.Tensor] = None
    
    gradRho: Optional[torch.Tensor] = None
    gradRhoL: Optional[torch.Tensor] = None
    
    # Ghost terms
    ghostIndices: Optional[torch.Tensor] = None
    ghostOffsets: Optional[torch.Tensor] = None

    # Free Surface terms
    surfaceMask: Optional[torch.Tensor] = None
    surfaceNormals: Optional[torch.Tensor] = None

    # CRK Kernel Correction Terms
    A: Optional[torch.Tensor] = None
    B: Optional[torch.Tensor] = None
    gradA: Optional[torch.Tensor] = None
    gradB: Optional[torch.Tensor] = None

    # Kernel Gradient Renormalization
    gradCorrectionMatrices: Optional[torch.Tensor] = None

    # grad-H Term:
    omega: Optional[torch.Tensor] = None
    apparentArea: Optional[torch.Tensor] = None

    # # Neighborhood
    numNeighbors: Optional[torch.Tensor] = None

    # Shifting Terms
    shiftVectors: Optional[torch.Tensor] = None

    # adjacency: Optional[SparseCOO] = None
    # species: Optional[torch.Tensor] = None

    # def _replace(self, **kwargs):
    #     return dataclasses.replace(self, **kwargs)
    
    def initializeNewState(self, scheme: str = 'momentum'):
        # print(f'Initializing new state for {scheme} {"momentum" in scheme.lower()}')
        return WeaklyCompressibleState(
            positions = self.positions.clone(),
            supports = self.supports.clone(),
            masses = self.masses.clone(),
            densities = self.densities.clone(),
            velocities = self.velocities.clone(),
            
            pressures = self.pressures.clone(),
            soundspeeds = self.soundspeeds.clone(),

            kinds = self.kinds.clone(),
            materials = self.materials.clone(),
            UIDs = self.UIDs.clone(),
            
            UIDcounter = self.UIDcounter,
            
            covarianceMatrices= None,
            # gradCorrectionMatrices = None,
            eigenValues = None,

            gradRho = None,
            gradRhoL = None,

            surfaceMask = None,
            surfaceNormals = None,

            A = None,
            B = None,
            gradA = None,
            gradB = None,
            gradCorrectionMatrices = None,
            omega = None,
            apparentArea = None,

            numNeighbors=None,

            ghostIndices = cloneOptionalTensor(self.ghostIndices),
            ghostOffsets = cloneOptionalTensor(self.ghostOffsets),

            shiftVectors=None,
            
            # species = self.species.clone() if self.species is not None else None,
        )
    def nograd(self):
        return WeaklyCompressibleState(
            positions= self.positions.detach().clone(),
            supports= self.supports.detach().clone(),
            masses= self.masses.detach().clone(),
            densities= self.densities.detach().clone(),
            velocities= self.velocities.detach().clone(),
            pressures= self.pressures.detach().clone(),
            soundspeeds= self.soundspeeds.detach().clone(),
            kinds= self.kinds.detach().clone(),
            materials= self.materials.detach().clone(),
            UIDs= self.UIDs.detach().clone(),
            UIDcounter= self.UIDcounter,

            covarianceMatrices=detachOptionalTensor(self.covarianceMatrices),
            # gradCorrectionMatrices=detachOptionalTensor(self.gradCorrectionMatrices),
            eigenValues=detachOptionalTensor(self.eigenValues),
            gradRho=detachOptionalTensor(self.gradRho),
            gradRhoL=detachOptionalTensor(self.gradRhoL),
            surfaceMask=detachOptionalTensor(self.surfaceMask),
            surfaceNormals=detachOptionalTensor(self.surfaceNormals),

            A = detachOptionalTensor(self.A),
            B = detachOptionalTensor(self.B),
            gradA = detachOptionalTensor(self.gradA),
            gradB = detachOptionalTensor(self.gradB),
            gradCorrectionMatrices=detachOptionalTensor(self.gradCorrectionMatrices),
            omega=detachOptionalTensor(self.omega),

            apparentArea=detachOptionalTensor(self.apparentArea),
            numNeighbors=detachOptionalTensor(self.numNeighbors),
            ghostIndices=detachOptionalTensor(self.ghostIndices),
            ghostOffsets=detachOptionalTensor(self.ghostOffsets),
            shiftVectors=detachOptionalTensor(self.shiftVectors),
        )
        
    def clone(self):
        return WeaklyCompressibleState(
            positions= self.positions.clone(),
            supports= self.supports.clone(),
            masses= self.masses.clone(),
            densities= self.densities.clone(),
            velocities= self.velocities.clone(),
            pressures= self.pressures.clone(),
            soundspeeds= self.soundspeeds.clone(),
            kinds= self.kinds.clone(),
            materials= self.materials.clone(),
            UIDs= self.UIDs.clone(),
            UIDcounter= self.UIDcounter,

            covarianceMatrices=cloneOptionalTensor(self.covarianceMatrices),
            # gradCorrectionMatrices=detachOptionalTensor(self.gradCorrectionMatrices),
            eigenValues=cloneOptionalTensor(self.eigenValues),
            gradRho=cloneOptionalTensor(self.gradRho),
            gradRhoL=cloneOptionalTensor(self.gradRhoL),
            surfaceMask=cloneOptionalTensor(self.surfaceMask),
            surfaceNormals=cloneOptionalTensor(self.surfaceNormals),

            A = cloneOptionalTensor(self.A),
            B = cloneOptionalTensor(self.B),
            gradA = cloneOptionalTensor(self.gradA),
            gradB = cloneOptionalTensor(self.gradB),
            gradCorrectionMatrices=cloneOptionalTensor(self.gradCorrectionMatrices),
            omega=cloneOptionalTensor(self.omega),

            apparentArea=cloneOptionalTensor(self.apparentArea),
            numNeighbors=cloneOptionalTensor(self.numNeighbors),
            ghostIndices=cloneOptionalTensor(self.ghostIndices),
            ghostOffsets=cloneOptionalTensor(self.ghostOffsets),
            shiftVectors=cloneOptionalTensor(self.shiftVectors),
        )
@dataclass(slots=True)
class WeaklyCompressibleUpdate:
    positions: torch.Tensor
    velocities: torch.Tensor
    densities: Optional[torch.Tensor] = None
    passive: Optional[torch.Tensor] = None
