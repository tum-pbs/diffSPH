from typing import Union, Optional
from dataclasses import dataclass
import torch

from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.kernels import SPHKernel, KernelType
from sphMath.operations import sph_op
from sphMath.modules.compressible import CompressibleState
from sphMath.sphOperations.shared import getTerms, compute_xij, scatter_sum
from sphMath.modules.switches.CullenDehnen2010 import computeShearTensor, computeM
from sphMath.modules.compressible import verbosePrint


from sphMath.modules.switches.Balsara1995 import computeBalsara1995Switch
from sphMath.modules.switches.Colagrossi2004 import computeColagrossi2004Switch
from sphMath.modules.switches.MorrisMonaghan1997 import computeMorrisMonaghan1997Switch
from sphMath.modules.switches.Rosswog2000 import computeRosswog2000Switch
from sphMath.modules.switches.CullenDehnen2010 import computeCullenTerms, computeCullenUpdate
from sphMath.modules.switches.CullenHopkins import computeHopkinsTerms, computeHopkinsUpdate
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple
from sphMath.util import getSetConfig

from enum import Enum
from sphMath.enums import ViscositySwitch


def computeViscositySwitch(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        dt: float = 0.0):
    
    viscositySwitch = getSetConfig(config, 'diffusionSwitch', 'scheme', ViscositySwitch.NoneSwitch)
    if viscositySwitch == ViscositySwitch.CullenDehnen2010:
        return computeCullenTerms(particles, kernel, neighborhood, supportScheme, config, dt)
    elif viscositySwitch == ViscositySwitch.CullenHopkins:
        return computeHopkinsTerms(particles, kernel, neighborhood, supportScheme, config, dt)
    elif viscositySwitch == ViscositySwitch.Balsara1995:
        return computeBalsara1995Switch(particles, kernel, neighborhood, supportScheme, config, dt)
    elif viscositySwitch == ViscositySwitch.Colagrossi2004:
        return computeColagrossi2004Switch(particles, kernel, neighborhood, supportScheme, config, dt)
    elif viscositySwitch == ViscositySwitch.MorrisMonaghan1997:
        return computeMorrisMonaghan1997Switch(particles, kernel, neighborhood, supportScheme, config, dt)
    elif viscositySwitch == ViscositySwitch.Rosswog2000:
        return computeRosswog2000Switch(particles, kernel, neighborhood, supportScheme, config, dt)
    else:
        return torch.ones_like(particles.densities), None
    
def updateViscositySwitch(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        dt: float = 0.0,
        dvdt : Optional[torch.Tensor] = None,
        switchState: Optional[torch.Tensor] = None
):
    
    viscositySwitch = getSetConfig(config, 'diffusionSwitch', 'scheme', ViscositySwitch.NoneSwitch)
    if viscositySwitch == ViscositySwitch.CullenDehnen2010:
        return computeCullenUpdate(particles, kernel, neighborhood, supportScheme, config, dt, dvdt, switchState)
    elif viscositySwitch == ViscositySwitch.CullenHopkins:
        return computeHopkinsUpdate(particles, kernel, neighborhood, supportScheme, config, dt, dvdt, switchState)
    elif viscositySwitch in [ViscositySwitch.Colagrossi2004, ViscositySwitch.MorrisMonaghan1997, ViscositySwitch.Rosswog2000]:
        return particles.alpha0s + dt * switchState, None
    else:
        return particles.alphas, None