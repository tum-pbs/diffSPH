
from sphMath.operations import sph_operation, mod

from sphMath.schemes.monaghanPrice import getPrice2007Config, MonaghanScheme, getMonaghan1997Config, getMonaghan1992Config, getMonaghanGingold1983Config, CompressibleSystem
from sphMath.schemes.compSPH import compSPHScheme, getCompSPHConfig, CompSPHSystem
from sphMath.schemes.PressureEnergySPH import PressureEnergyScheme, getPressureEnergyConfig
from sphMath.schemes.CRKSPH import CRKScheme, getCRKConfig
# from sphMath.schemes.deltaSPH import deltaPlusSPHScheme, getDeltaSPHConfig



from sphMath.integration import *
from sphMath.kernels import *
from sphMath.neighborhood import DomainDescription
from sphMath.sampling import buildDomainDescription
from typing import Optional
# from sphMath.modules.viscositySwitch import ViscositySwitch
from sphMath.modules.compSPH import EnergyScheme

from enum import Enum
import torch
from sphMath.enums import *


from sphMath.schemes.deltaSPH import deltaPlusSPHScheme, DeltaPlusSPHSystem, getDeltaSPHConfig

def getSimulationScheme(
    schema: SimulationScheme,
    kernel: KernelType,
    integrator: IntegrationSchemeType,
    
    gamma: float = 5./3.,
    targetNeighbors: int = 50,
    domain: DomainDescription = buildDomainDescription(1, 2, True, torch.device('cpu'), torch.float32),
    viscositySwitch: Optional[ViscositySwitch] = None,
    supportScheme: AdaptiveSupportScheme = AdaptiveSupportScheme.OwenScheme,
    energyScheme: Optional[EnergyScheme] = None,
    verletScale: float = 1.4
    ):
    wrappedKernel = kernel
    integrationScheme = getIntegrator(integrator)
    
    if schema == SimulationScheme.CompSPH:
        scheme = compSPHScheme
        schemeSystem = CompSPHSystem
        schemeConfig_fn = getCompSPHConfig
    elif schema == SimulationScheme.PESPH:
        scheme = PressureEnergyScheme
        schemeSystem = CompressibleSystem
        schemeConfig_fn = getPressureEnergyConfig
    elif schema == SimulationScheme.CRKSPH:
        scheme = CRKScheme
        schemeSystem = CompSPHSystem   
        schemeConfig_fn = getCRKConfig
    elif schema == SimulationScheme.Price2007:
        scheme = MonaghanScheme
        schemeSystem = CompressibleSystem
        schemeConfig_fn = getPrice2007Config
    elif schema == SimulationScheme.Monaghan1997:
        scheme = MonaghanScheme
        schemeSystem = CompressibleSystem
        schemeConfig_fn = getMonaghan1997Config
    elif schema == SimulationScheme.Monaghan1992:
        scheme = MonaghanScheme
        schemeSystem = CompressibleSystem
        schemeConfig_fn = getMonaghan1992Config
    elif schema == SimulationScheme.MonaghanGingold1983:
        scheme = MonaghanScheme
        schemeSystem = CompressibleSystem
        schemeConfig_fn = getMonaghanGingold1983Config
    elif schema == SimulationScheme.DeltaSPH:
        scheme = deltaPlusSPHScheme
        schemeSystem = DeltaPlusSPHSystem
        schemeConfig_fn = getDeltaSPHConfig
    else:
        raise ValueError(f"Unknown schema {schema}")
    
    # print(schema)
    
    schemeConfig = schemeConfig_fn(gamma = gamma, kernel = wrappedKernel, targetNeighbors = targetNeighbors, domain = domain, verletScale=verletScale)
    
    if viscositySwitch is not None:
        if 'diffusionSwitch' in schemeConfig:
            schemeConfig['diffusionSwitch']['scheme'] = viscositySwitch
        else:
            schemeConfig['diffusionSwitch'] = {'scheme': viscositySwitch}
    if supportScheme is not None:
        if 'support' in schemeConfig:
            schemeConfig['support']['scheme'] = supportScheme
        else:
            schemeConfig['support'] = {'scheme': supportScheme}
    if energyScheme is not None:
        schemeConfig['energyScheme'] = energyScheme.name
        
    schemaName = schemeConfig['schemeName']
    if 'diffusionSwitch' in schemeConfig:
        if 'scheme' in schemeConfig['diffusionSwitch']:
            if schemeConfig['diffusionSwitch']['scheme'] == ViscositySwitch.CullenDehnen2010:
                schemaName += ' - Cullen'
            elif schemeConfig['diffusionSwitch']['scheme'] == ViscositySwitch.CullenHopkins:
                schemaName += ' - Cullen[Hopkins]'
            elif schemeConfig['diffusionSwitch']['scheme'] is not None:
                schemaName += ' - ' + schemeConfig['diffusionSwitch']['scheme'].name
    if schema == SimulationScheme.CompSPH or schema == SimulationScheme.CRKSPH:
        schemaName += ' - E=' + schemeConfig['energyScheme'].name
    schemaName += ' - ' + kernel.name + ' - ' + integrator.name
    
    schemeConfig['schemeName'] = schemaName
    schemeConfig['scheme'] = schema
    schemeConfig['integrationScheme'] = integrator
    schemeConfig['kernel'] = wrappedKernel
    
        
    return scheme, schemeSystem, schemeConfig, integrationScheme