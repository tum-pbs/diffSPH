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

@torch.jit.script
@dataclass(slots=True)
class BasicState:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor    
    velocities: torch.Tensor
    
    # meta properties for particles
    kinds: torch.Tensor # 0 for fluid, 1 for boundary, 2 for ghost
    materials: torch.Tensor # specific subsets of particles, also indicates body id
    UIDs : torch.Tensor # unique identifiers for particles, ghost particles have negative UIDs
    UIDcounter: int = 0

    # CRK Kernel Correction Terms
    A: Optional[torch.Tensor] = None
    B: Optional[torch.Tensor] = None
    gradA: Optional[torch.Tensor] = None
    gradB: Optional[torch.Tensor] = None

    # Kernel Gradient Renormalization
    gradCorrectionMatrices: Optional[torch.Tensor] = None

    # grad-H Term:
    omega: Optional[torch.Tensor] = None

    apparentArea : Optional[torch.Tensor] = None

    # Ghost terms
    ghostIndices: Optional[torch.Tensor] = None
    ghostOffsets: Optional[torch.Tensor] = None

    # Neighborhood terms
    numNeighbors: Optional[torch.Tensor] = None

    def initializeNewState(self, scheme : str = 'internalEnergy'):

        return BasicState(
            positions = self.positions.clone(),
            supports = self.supports.clone(),
            masses = self.masses.clone(),
            densities = self.densities.clone(),
            velocities = self.velocities.clone(),
            
            kinds = self.kinds.clone(),
            materials = self.materials.clone(),
            UIDs = self.UIDs.clone(),
            UIDcounter= self.UIDcounter,

            A = None,
            B = None,
            gradA = None,
            gradB = None,
            gradCorrectionMatrices = None,
            omega = None,
            apparentArea=None,
            numNeighbors=None,

            ghostIndices=None,
            ghostOffsets=None
            
            # species = self.species.clone() if self.species is not None else None,
        )
    
    def nograd(self):
        return BasicState(
            positions = self.positions.detach().clone(),
            supports = self.supports.detach().clone(),
            masses = self.masses.detach().clone(),
            densities = self.densities.detach().clone(),
            velocities = self.velocities.detach().clone(),

            kinds = self.kinds.detach().clone(),
            materials = self.materials.detach().clone(),
            UIDs = self.UIDs.detach().clone(),
            UIDcounter= self.UIDcounter,

            ghostIndices = detachOptionalTensor(self.ghostIndices),
            ghostOffsets = detachOptionalTensor(self.ghostOffsets),
            apparentArea= detachOptionalTensor(self.apparentArea),
            # omega = detachOptionalTensor(self.omega),
            A=detachOptionalTensor(self.A),
            B=detachOptionalTensor(self.B),
            gradA=detachOptionalTensor(self.gradA),
            gradB=detachOptionalTensor(self.gradB),
            gradCorrectionMatrices=detachOptionalTensor(self.gradCorrectionMatrices),
            omega=detachOptionalTensor(self.omega),
            numNeighbors = detachOptionalTensor(self.numNeighbors),
        )
    def clone(self):
        return BasicState(
            positions = self.positions.clone(),
            supports = self.supports.clone(),
            masses = self.masses.clone(),
            densities = self.densities.clone(),
            velocities = self.velocities.clone(),

            kinds = self.kinds.clone(),
            materials = self.materials.clone(),
            UIDs = self.UIDs.clone(),
            UIDcounter= self.UIDcounter,

            ghostIndices = cloneOptionalTensor(self.ghostIndices),
            ghostOffsets = cloneOptionalTensor(self.ghostOffsets),
            apparentArea= cloneOptionalTensor(self.apparentArea),
            # omega = detachOptionalTensor(self.omega),
            A=cloneOptionalTensor(self.A),
            B=cloneOptionalTensor(self.B),
            gradA=cloneOptionalTensor(self.gradA),
            gradB=cloneOptionalTensor(self.gradB),
            gradCorrectionMatrices=cloneOptionalTensor(self.gradCorrectionMatrices),
            omega=cloneOptionalTensor(self.omega),
            numNeighbors = cloneOptionalTensor(self.numNeighbors),
        )