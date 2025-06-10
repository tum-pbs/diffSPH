import torch
from typing import Optional

def idealGasEOS(A: Optional[torch.Tensor], u: Optional[torch.Tensor], P: Optional[torch.Tensor], 
             rho: torch.Tensor, gamma: float):
    given = (1 if A is not None else 0) + (1 if u is not None else 0) + (1 if P is not None else 0)
    # if given < 2:
        # raise ValueError('At least two of the three parameters must be given')
    
    P_, u_, A_, rho_ = P, u, A, rho
    
    c_s = None
    if u is not None:
        c_s = torch.sqrt(u.abs() * gamma * (gamma - 1))
    elif P is not None:
        c_s = torch.sqrt(gamma * P.abs() / rho)
    elif A is not None:
        c_s = torch.sqrt(gamma * rho ** (gamma - 1) * A)

    if P is None and u is not None:
        P_ = (gamma - 1) * rho * u
    elif P is None and A is not None:
        P_ = A * rho**gamma
    
    if u is None and A is not None:
        u_ = A * rho**(gamma - 1) / (gamma - 1)
    elif u is None and P is not None:
        u_ = P / rho / (gamma - 1)

    if A is None and u is not None:
        A_ = u * (gamma - 1) * rho**(1 - gamma)
    elif A is None and P is not None:
        A_ = P / rho**gamma

    return A_, u_, P_, c_s






def stiffTaitEOS(rho, rho0: float, c_s: float, polytropicExponent: float):

    return rho0 * c_s**2 / polytropicExponent * ((rho / rho0)**polytropicExponent - 1)

def TaitEOS(rho, rho0: float, kappa: float, config):
    return kappa * (rho - rho0)

def isoThermalEOS(rho, rho0: float, c_s: float):
    return c_s**2 * (rho - rho0)

def polytropicEOS(rho, polytropicExponent : float, kappa : float):
    return kappa * (rho)**polytropicExponent

def murnaghanEOS(rho, rho0: float, kappa: float, exponent: float):
    return kappa / exponent * ((rho / rho0)**exponent - 1)

from torch.profiler import record_function
from sphMath.util import getSetConfig


def computeEOS_WC(particles, config):
    rho0 = config['fluid']['rho0']
    c_s = config['fluid']['c_s']
    
    eosType = getSetConfig(config, 'EOS', 'type', 'isoThermal')
    kappa = getSetConfig(config, 'EOS', 'kappa', 1.3)
    polytropicExponent = getSetConfig(config, 'EOS', 'polytropicExponent', 7)
    gas_constant = getSetConfig(config, 'EOS', 'gas_constant', 8.314)
    molarMass = getSetConfig(config, 'EOS', 'molarMass', 0.02897)
        
    with record_function("[SPH] - Equation of State"):
        if eosType == 'stiffTait':
            return stiffTaitEOS(particles.densities, rho0, c_s, polytropicExponent)
        elif eosType == 'Tait':
            return TaitEOS(particles.densities, rho0, kappa, config)
        elif eosType == 'isoThermal':
            return isoThermalEOS(particles.densities, rho0, c_s)
        elif eosType == 'polytropic':
            return polytropicEOS(particles.densities, polytropicExponent, kappa)
        elif eosType == 'murnaghan':
            return murnaghanEOS(particles.densities, rho0, kappa, polytropicExponent)
        else:
            raise ValueError('EOS type not recognized')
