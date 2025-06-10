from sphMath.integrationSchemes.util import IntegrationScheme, updateStateEuler, updateStateSemiImplicitEuler
from sphMath.integrationSchemes.euler import integrateExplicitEuler, integrateSemiImplicitEuler
from sphMath.integrationSchemes.butcher import forwardEuler, RungeKutta2, midPoint, heunsMethod, ralston2nd, RungeKutta3, heunsMethod3rd, ralston3rd, Wray3rd, SSPRK3, RungeKutta4, RungeKutta4alt, Nystrom5th, EPEC, EPECmodified
from sphMath.integrationSchemes.verlet import leapFrog, symplecticEuler, velocityVerlet
from sphMath.integrationSchemes.tvd import TVDRK3, TVDRK2
from sphMath.integrationSchemes.ruth import PEFRL, VEFRL

semiImplicitEuler = lambda state, dt, f, *args, **kwargs: integrateSemiImplicitEuler(state, dt, f, *args, **kwargs)
explicitEuler = lambda state, dt, f, *args, **kwargs: integrateExplicitEuler(state, dt, f, *args, **kwargs)

IntegrationSchemes = []

from sphMath.integrationSchemes.util import IntegrationSchemeType

IntegrationSchemes.append(IntegrationScheme(forwardEuler, 'Forward Euler', IntegrationSchemeType.forwardEuler, 1, True, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta2,  'Midpoint',      IntegrationSchemeType.rungeKutta2, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(heunsMethod,  'Heun\'s Method (2nd order)', IntegrationSchemeType.heunsMethod, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(ralston2nd,   'Ralston\'s Method (2nd order)', IntegrationSchemeType.ralston2nd, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta3,  'RK3',           IntegrationSchemeType.rungeKutta3, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(heunsMethod3rd, 'Heun\'s Method (3rd order)', IntegrationSchemeType.heunsMethod3rd, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(ralston3rd,   'Ralston\'s Method (3rd order)', IntegrationSchemeType.ralston3rd, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(Wray3rd,      'Wray\'s Method (3rd order)', IntegrationSchemeType.wray3rd, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK3,       'SSP RK3',       IntegrationSchemeType.sspRK3, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta4,  'RK4',           IntegrationSchemeType.rungeKutta4, 4, True, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta4alt, 'RK4 (alternative)', IntegrationSchemeType.rungeKutta4alt, 4, True, True))
IntegrationSchemes.append(IntegrationScheme(Nystrom5th,   'Nystrom 5th order', IntegrationSchemeType.nystrom5th, 5, True, True))
IntegrationSchemes.append(IntegrationScheme(leapFrog, 'Leap Frog', IntegrationSchemeType.leapFrog, 2, False, False))
IntegrationSchemes.append(IntegrationScheme(symplecticEuler, 'Symplectic Euler', IntegrationSchemeType.symplecticEuler, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(velocityVerlet, 'Velocity Verlet', IntegrationSchemeType.velocityVerlet, 2, False, False))
IntegrationSchemes.append(IntegrationScheme(PEFRL, 'PEFRL', IntegrationSchemeType.pefrl, 4, False, False))
IntegrationSchemes.append(IntegrationScheme(VEFRL, 'VEFRL', IntegrationSchemeType.vefrl, 4, True, True))
IntegrationSchemes.append(IntegrationScheme(EPEC, 'EPEC', IntegrationSchemeType.epec, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(EPECmodified, 'EPEC Modified', IntegrationSchemeType.epecModified, 2, False, False))
IntegrationSchemes.append(IntegrationScheme(TVDRK3, 'TVD RK3', IntegrationSchemeType.tvdRK3, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(TVDRK2, 'TVD RK2', IntegrationSchemeType.tvdRK2, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(lambda state, dt, f, *args, **kwargs: integrateSemiImplicitEuler(state, dt, f, *args, **kwargs), 'Semi-Implicit Euler', IntegrationSchemeType.semiImplicitEuler, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(lambda state, dt, f, *args, **kwargs: integrateExplicitEuler(state, dt, f, *args, **kwargs), 'Explicit Euler', IntegrationSchemeType.explicitEuler, 1, True, True))

def getPreferredScheme(order):
    if order == 1:
        return semiImplicitEuler
    elif order == 2:
        return symplecticEuler
    elif order == 3:
        return TVDRK3
    elif order == 4:
        return RungeKutta4
    elif order == 5:
        return Nystrom5th
    else:
        raise ValueError(f"No scheme for order {order}")

def getIntegrator(integrator: str):
    for scheme in IntegrationSchemes:
        if scheme.name == integrator or scheme.identifier == integrator:
            return scheme
    raise ValueError(f"Unknown integrator {integrator}")

def getIntegrationEnum(integrator: str):
    for scheme in IntegrationSchemes:
        if scheme.name == integrator or scheme.identifier.name == integrator:
            return scheme.identifier
    raise ValueError(f"Unknown integrator {integrator}")