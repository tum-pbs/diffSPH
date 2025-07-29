import torch
from enum import Enum



from enum import Enum

class SupportScheme(Enum):
    Gather = 1
    Scatter = 2
    Symmetric = 3
    SuperSymmetric = 4
    
class Operation(Enum):
    Interpolate = 1
    Gradient = 2
    Divergence = 3
    Laplacian = 4
    Curl = 5
    Density = 6

class GradientMode(Enum):
    Naive = 1
    Difference = 2
    Summation = 3
    Symmetric = 4

class DivergenceMode(Enum):
    div = 1
    dot = 2

class LaplacianMode(Enum):
    naive = 1
    Brookshaw = 2
    dot = 6
    default = 7


@torch.jit.script
class AdaptiveSupportScheme(Enum):
    NoScheme = 0
    OwenScheme = 1
    MonaghanScheme = 2
    
@torch.jit.script
class SimulationScheme(Enum):
    CompSPH = 0
    PESPH = 1
    CRKSPH = 2
    Price2007 = 3
    Monaghan1997 = 4
    Monaghan1992 = 5
    MonaghanGingold1983 = 6
    DeltaSPH = 7
    
    
@torch.jit.script
class KernelType(Enum):
    Poly6 = 0
    CubicSpline = 1
    QuarticSpline = 2
    QuinticSpline = 3
    B7 = 4
    Wendland2 = 5
    Wendland4 = 6
    Wendland6 = 7
    Spiky = 8
    ViscosityKernel = 9
    CohesionKernel = 10
    AdhesionKernel = 11
    
    
@torch.jit.script
class ViscositySwitch(Enum):
    Balsara1995 = 0
    Colagrossi2004 = 1
    CullenDehnen2010 = 2
    CullenHopkins = 3
    MorrisMonaghan1997 = 4
    Rosswog2000 = 5
    NoneSwitch = 6
    
    
@torch.jit.script
class IntegrationSchemeType(Enum):
    forwardEuler = 0
    rungeKutta2 = 1
    heunsMethod = 2
    ralston2nd = 3
    rungeKutta3 = 4
    heunsMethod3rd = 5
    ralston3rd = 6
    wray3rd = 7
    sspRK3 = 8
    rungeKutta4 = 9
    rungeKutta4alt = 10
    nystrom5th = 11
    leapFrog = 12
    symplecticEuler = 13
    velocityVerlet = 14
    pefrl = 15
    vefrl = 16
    epec = 17
    epecModified = 18
    tvdRK3 = 19
    tvdRK2 = 20
    semiImplicitEuler = 21
    explicitEuler = 22
    
    
    
@torch.jit.script
class EnergyScheme(Enum):
    equalWork = 0
    PdV = 1
    diminishing = 2
    monotonic = 3
    hybrid = 4
    CRK = 5


@torch.jit.script
class KernelCorrectionScheme(Enum):
    NoCorrection = 0
    CRKSPH = 1
    gradientRenorm = 2
    gradH = 3

