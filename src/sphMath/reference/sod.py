import numpy as np
import scipy
import scipy.optimize


def sound_speed(gamma, pressure, density, dustFrac=0.):
    """
    Calculate sound speed, scaled by the dust fraction according to:

        .. math::
            \widetilde{c}_s = c_s \sqrt{1 - \epsilon}

    Where :math:`\epsilon` is the dustFrac
    """
    scale = np.sqrt(1 - dustFrac)
    return np.sqrt(gamma * pressure / density) * scale


def shock_tube_function(p4, p1, p5, rho1, rho5, gamma, dustFrac=0.):
    """
    Shock tube equation
    """
    z = (p4 / p5 - 1.)
    c1 = sound_speed(gamma, p1, rho1, dustFrac)
    c5 = sound_speed(gamma, p5, rho5, dustFrac)

    gm1 = gamma - 1.
    gp1 = gamma + 1.
    g2 = 2. * gamma

    fact = gm1 / g2 * (c5 / c1) * z / np.sqrt(1. + gp1 / g2 * z)
    fact = (1. - fact) ** (g2 / gm1)

    return p1 * fact - p4


def calculate_regions(pl, ul, rhol, pr, ur, rhor, gamma=1.4, dustFrac=0.):
    """
    Compute regions
    :rtype : tuple
    :return: returns p, rho and u for regions 1,3,4,5 as well as the shock speed
    """
    # if pl > pr...
    rho1 = rhol
    p1 = pl
    u1 = ul
    rho5 = rhor
    p5 = pr
    u5 = ur

    # unless...
    if pl < pr:
        rho1 = rhor
        p1 = pr
        u1 = ur
        rho5 = rhol
        p5 = pl
        u5 = ul

    # solve for post-shock pressure
    p4 = scipy.optimize.fsolve(shock_tube_function, p1, (p1, p5, rho1, rho5, gamma))[0]

    # compute post-shock density and velocity
    z = (p4 / p5 - 1.)
    c5 = sound_speed(gamma, p5, rho5, dustFrac)

    gm1 = gamma - 1.
    gp1 = gamma + 1.
    gmfac1 = 0.5 * gm1 / gamma
    gmfac2 = 0.5 * gp1 / gamma

    fact = np.sqrt(1. + gmfac2 * z)

    u4 = c5 * z / (gamma * fact)
    rho4 = rho5 * (1. + gmfac2 * z) / (1. + gmfac1 * z)

    # shock speed
    w = c5 * fact

    # compute values at foot of rarefaction
    p3 = p4
    u3 = u4
    rho3 = rho1 * (p3 / p1) ** (1. / gamma)
    return (p1, rho1, u1), (p3, rho3, u3), (p4, rho4, u4), (p5, rho5, u5), w


def calc_positions(pl, pr, region1, region3, w, xi, t, gamma, dustFrac=0.):
    """
    :return: tuple of positions in the following order ->
            Head of Rarefaction: xhd,  Foot of Rarefaction: xft,
            Contact Discontinuity: xcd, Shock: xsh
    """
    p1, rho1 = region1[:2]  # don't need velocity
    p3, rho3, u3 = region3
    c1 = sound_speed(gamma, p1, rho1, dustFrac)
    c3 = sound_speed(gamma, p3, rho3, dustFrac)

    if pl > pr:
        xsh = xi + w * t
        xcd = xi + u3 * t
        xft = xi + (u3 - c3) * t
        xhd = xi - c1 * t
    else:
        # pr > pl
        xsh = xi - w * t
        xcd = xi - u3 * t
        xft = xi - (u3 - c3) * t
        xhd = xi + c1 * t

    return xhd, xft, xcd, xsh


def region_states(pl, pr, region1, region3, region4, region5):
    """
    :return: dictionary (region no.: p, rho, u), except for rarefaction region
    where the value is a string, obviously
    """
    if pl > pr:
        return {'Region 1': region1,
                'Region 2': 'RAREFACTION',
                'Region 3': region3,
                'Region 4': region4,
                'Region 5': region5}
    else:
        return {'Region 1': region5,
                'Region 2': region4,
                'Region 3': region3,
                'Region 4': 'RAREFACTION',
                'Region 5': region1}


def create_arrays(pl, pr, xl, xr, positions, state1, state3, state4, state5,
                  npts, gamma, t, xi, dustFrac=0.):
    """
    :return: tuple of x, p, rho and u values across the domain of interest
    """
    xhd, xft, xcd, xsh = positions
    p1, rho1, u1 = state1
    p3, rho3, u3 = state3
    p4, rho4, u4 = state4
    p5, rho5, u5 = state5
    gm1 = gamma - 1.
    gp1 = gamma + 1.

    x_arr = np.linspace(xl, xr, npts)
    rho = np.zeros(npts, dtype=float)
    p = np.zeros(npts, dtype=float)
    u = np.zeros(npts, dtype=float)
    c1 = sound_speed(gamma, p1, rho1, dustFrac)
    if pl > pr:
        for i, x in enumerate(x_arr):
            if x < xhd:
                rho[i] = rho1
                p[i] = p1
                u[i] = u1
            elif x < xft:
                u[i] = 2. / gp1 * (c1 + (x - xi) / t)
                fact = 1. - 0.5 * gm1 * u[i] / c1
                rho[i] = rho1 * fact ** (2. / gm1)
                p[i] = p1 * fact ** (2. * gamma / gm1)
            elif x < xcd:
                rho[i] = rho3
                p[i] = p3
                u[i] = u3
            elif x < xsh:
                rho[i] = rho4
                p[i] = p4
                u[i] = u4
            else:
                rho[i] = rho5
                p[i] = p5
                u[i] = u5
    else:
        for i, x in enumerate(x_arr):
            if x < xsh:
                rho[i] = rho5
                p[i] = p5
                u[i] = -u1
            elif x < xcd:
                rho[i] = rho4
                p[i] = p4
                u[i] = -u4
            elif x < xft:
                rho[i] = rho3
                p[i] = p3
                u[i] = -u3
            elif x < xhd:
                u[i] = -2. / gp1 * (c1 + (xi - x) / t)
                fact = 1. + 0.5 * gm1 * u[i] / c1
                rho[i] = rho1 * fact ** (2. / gm1)
                p[i] = p1 * fact ** (2. * gamma / gm1)
            else:
                rho[i] = rho1
                p[i] = p1
                u[i] = -u1

    return x_arr, p, rho, u


def solve(left_state, right_state, geometry, t, gamma=1.4, npts=500,
          dustFrac=0.):
    """
    Solves the Sod shock tube problem (i.e. riemann problem) of discontinuity
    across an interface.

    Parameters
    ----------
    left_state, right_state: tuple
        A tuple of the state (pressure, density, velocity) on each side of the
        shocktube barrier for the ICs.  In the case of a dusty-gas, the density
        should be the gas density.
    geometry: tuple
        A tuple of positions for (left boundary, right boundary, barrier)
    t: float
        Time to calculate the solution at
    gamma: float
        Adiabatic index for the gas.
    npts: int
        number of points for array of pressure, density and velocity
    dustFrac: float
        Uniform fraction for the gas, between 0 and 1.

    Returns
    -------
    positions: dict
        Locations of the important places (rarefaction wave, shock, etc...)
    regions: dict
        constant pressure, density and velocity states in distinct regions
    values: dict
        Arrays of pressure, density, and velocity as a function of position.
        The density ('rho') is the gas density, which may differ from the
        total density in a dusty-gas.
        Also calculates the specific internal energy
    """

    pl, rhol, ul = left_state
    pr, rhor, ur = right_state
    xl, xr, xi = geometry

    # basic checking
    if xl >= xr:
        print('xl has to be less than xr!')
        exit()
    if xi >= xr or xi <= xl:
        print('xi has in between xl and xr!')
        exit()

    # calculate regions
    region1, region3, region4, region5, w = \
        calculate_regions(pl, ul, rhol, pr, ur, rhor, gamma, dustFrac)

    regions = region_states(pl, pr, region1, region3, region4, region5)

    # calculate positions
    x_positions = calc_positions(pl, pr, region1, region3, w, xi, t, gamma,
                                 dustFrac)

    pos_description = ('Head of Rarefaction', 'Foot of Rarefaction',
                       'Contact Discontinuity', 'Shock')
    positions = dict(zip(pos_description, x_positions))

    # create arrays
    x, p, rho, u = create_arrays(pl, pr, xl, xr, x_positions,
                                 region1, region3, region4, region5,
                                 npts, gamma, t, xi, dustFrac)

    energy = p / (rho * (gamma - 1.0))
    rho_total = rho / (1.0 - dustFrac)
    val_dict = {'x': x, 'p': p, 'rho': rho, 'u': u, 'energy': energy,
                'rho_total': rho_total}

    return positions, regions, val_dict


from sphMath.modules.eos import idealGasEOS
# from sphMath.modules.compressible import CompressibleParticleSet, SPHSystem, systemToParticles, systemUpdate

from sphMath.schemes.gasDynamics import CompressibleState, CompressibleSystem, CompressibleUpdate

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sphMath.operations import sph_operation, mod
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from sphMath.util import volumeToSupport
from sphMath.util import ParticleSet
from sphMath.sampling import buildDomainDescription, sampleRegularParticles

from sphMath.plotting import visualizeParticles
from sphMath.sampling import generateNoiseInterpolator
from sphMath.util import ParticleSetWithQuantity, mergeParticles
import random
from sphMath.sampling import getSpacing
from sphMath.operations import sph_op
from sphMath.sampling import generateTestData
from sphMath.modules.density import computeDensity, computeDensityGradient, computeRenormalizedDensityGradient, computeDensityDeltaTerm
from sphMath.reference.sod import solve

from typing import NamedTuple, Tuple
from sphMath.kernels import KernelType
# from sphMath.neighborhood import buildSuperSymmetricNeighborhood
from sphMath.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport
from sphMath.modules.adaptiveSmoothingASPH import n_h_to_nH


class sodInitialState(NamedTuple):
    p: float
    rho: float
    v: float

def decoratePlot(axis, positions, text = True):        
    axis.axvline(positions['Head of Rarefaction'], color='black', linestyle='--', alpha = 0.5)
    axis.axvline(positions['Foot of Rarefaction'], color='black', linestyle='--', alpha = 0.5)
    axis.axvline(positions['Contact Discontinuity'], color='black', linestyle='--', alpha = 0.5)
    axis.axvline(positions['Shock'], color='black', linestyle='--', alpha = 0.5)
    if text:
        ymax = axis.get_ylim()[1]
        ymin = axis.get_ylim()[0]
        position = ymin + 0.85 * (ymax - ymin)
        axis.text(positions['Head of Rarefaction'] / 2, position, 'I')
        axis.text((positions['Head of Rarefaction'] + positions['Foot of Rarefaction']) / 2, position, 'II')
        axis.text((positions['Foot of Rarefaction'] + positions['Contact Discontinuity']) / 2, position, 'III')
        axis.text((positions['Contact Discontinuity'] + positions['Shock']) / 2, position, 'IV')
        axis.text((positions['Shock'] + 1) / 2, position, 'V')
        
        
        
# from sphMath.modules.compressible import CompressibleParticleSet, SPHSystem, systemToParticles, systemUpdate
from sphMath.reference.sod import decoratePlot

        
from sphMath.reference.sod import decoratePlot

        
        

        
def plotSod(simulationState, simulationConfig, domain, gamma, leftState: sodInitialState, rightState: sodInitialState, plotReference = True, plotLabels = True, scatter = False, t_ = None):
    fig, axis = plt.subplots(2, 3, figsize=(10, 5), squeeze=False, sharex=True, sharey=False)

    referenceState = simulationState.systemState
    neighborhood, sparseNeighborhood = buildNeighborhood(referenceState, referenceState, domain, verletScale = 1.0)
    CSR = coo_to_csr(sparseNeighborhood)
    pos = referenceState.positions.cpu().numpy()
    densities = referenceState.densities.cpu().numpy()
    velocities = referenceState.velocities.cpu().numpy()
    thermalEnergy = referenceState.thermalEnergies.cpu().numpy() if hasattr(referenceState, 'thermalEnergies') else referenceState.internalEnergies.cpu().numpy()
    supports = referenceState.supports.cpu().numpy()
    pressures = referenceState.pressures.cpu().numpy() if hasattr(referenceState, 'pressures') and referenceState.pressures is not None else referenceState.P.cpu().numpy()
    neighbors = CSR.rowEntries.cpu().numpy()
    masses = referenceState.masses.cpu().numpy()
    
    A_, u_, P_, c_ = idealGasEOS(A = None, u = torch.tensor(thermalEnergy, dtype = referenceState.densities.dtype, device = referenceState.densities.device), P = None, rho = referenceState.densities, gamma = gamma)
    A_ = A_.cpu().numpy()
    
    kineticEnergy_ = 0.5 * (torch.linalg.norm(simulationState.systemState.velocities, dim = -1) **2 * simulationState.systemState.masses).sum()
    thermalEnergy_ = (simulationState.systemState.internalEnergies * simulationState.systemState.masses).sum()
    totalEnergy = kineticEnergy_ + thermalEnergy_
        

    t = (simulationState.t if simulationState.t > 0 else 1e-5) if t_ is None else t_

    indices = torch.argsort(referenceState.positions[:,0]).cpu().numpy()
    indices = indices[pos[indices][:,0] > 0]

    fig.suptitle(f'{simulationConfig["schemeName"]}\nnx = {pos.shape[0]//2}, t = {t:6.4g}, Kinetic = {kineticEnergy_.cpu().item():6.4g}, Thermal = {thermalEnergy_.cpu().item():6.4g}, Total = {totalEnergy.cpu().item():6.4g}')
    if not scatter:
        axis[0,0].plot(pos[indices], densities[indices], label='Density')
        axis[0,1].plot(pos[indices], supports[indices], label='Supports')
        axis[0,2].plot(pos[indices], velocities[indices], label='Velocity')
        axis[1,0].plot(pos[indices], thermalEnergy[indices], label='Thermal Energy')
        axis[1,1].plot(pos[indices], pressures[indices], label='Pressure')
        axis[1,2].plot(pos[indices], A_[indices], label='Neighbors')
    else:
        s = 1
        axis[0,0].scatter(pos[indices], densities[indices], s = s, label='Density')
        axis[0,1].scatter(pos[indices], supports[indices], s = s, label='Supports')
        axis[0,2].scatter(pos[indices], velocities[indices], s = s, label='Velocity')
        axis[1,0].scatter(pos[indices], thermalEnergy[indices], s = s, label='Thermal Energy')
        axis[1,1].scatter(pos[indices], pressures[indices], s = s, label='Pressure')
        axis[1,2].scatter(pos[indices], A_[indices], s = s, label='Masses')
    
    axis[0,0].set_title('Density')
    axis[0,1].set_title('Supports')
    axis[0,2].set_title('Velocity')
    axis[1,0].set_title('Thermal Energy')
    axis[1,1].set_title('Pressure')
    axis[1,2].set_title('Entropy')
    if plotReference:
        dustFrac = 0.0
        npts = 500
        # t = 1e-4
        left_state = (leftState.p,leftState.rho,leftState.v)
        right_state = (rightState.p, rightState.rho, rightState.v)
        # right_state = (0.1, 0.5, 0.)

        positions, regions, values = solve(left_state=left_state, \
            right_state=right_state, geometry=(0., 1., 0.5), t=t.cpu().item() if isinstance(t, torch.Tensor) else t,
            gamma=gamma, npts=npts, dustFrac=dustFrac)
        axis[0,0].plot(values['x'], values['rho'], label='Reference Solution', alpha = 0.5, ls = ':', c = 'black')
        axis[1,1].plot(values['x'], values['p'], label='Reference Solution', alpha = 0.5, ls = ':', c = 'black')
        axis[0,2].plot(values['x'], values['u'], label='Reference Solution', alpha = 0.5, ls = ':', c = 'black')
        axis[1,0].plot(values['x'], values['energy'], label='Reference Solution', alpha = 0.5, ls = ':', c = 'black')
            
        if plotLabels:
            decoratePlot(axis[0,0], positions, text = True)
            decoratePlot(axis[0,1], positions, text = False)
            decoratePlot(axis[0,2], positions, text = True)
            decoratePlot(axis[1,0], positions, text = False)
            decoratePlot(axis[1,1], positions, text = True)
            decoratePlot(axis[1,2], positions, text = False)

    fig.tight_layout()

from sphMath.schemes.gasDynamics import CompressibleState, CompressibleSystem
from sphMath.neighborhood import evaluateNeighborhood, SupportScheme, filterNeighborhoodByKind, filterNeighborhood

def generateSod1D(nx, samplingRatio, leftState, rightState, gamma, wrappedKernel, targetNeighbors, dtype = torch.float32, device = torch.device('cpu'), smoothIC = True, SimulationSystem = CompressibleSystem):
    
    domain = buildDomainDescription(2, 1, periodic = True, device = device, dtype = dtype)
    dim = 1
    actualRatio = nx / (nx // samplingRatio)
    particles_l = sampleRegularParticles(nx, buildDomainDescription(1, dim, periodic = True, device = device, dtype = dtype), targetNeighbors, jitter = 0.0)
    particles_r = sampleRegularParticles(nx // samplingRatio, buildDomainDescription(1, dim, periodic = True, device = device, dtype = dtype), targetNeighbors, jitter = 0.0)

    particles_l = particles_l._replace(positions = particles_l.positions - torch.tensor([0.0], device = device, dtype = dtype))
    particles_r = particles_r._replace(positions = particles_r.positions + torch.tensor([0.0], device = device, dtype = dtype))

    particles_r = particles_r._replace(masses = torch.ones_like(particles_r.masses) * particles_l.masses.min())

    pos_r = particles_r.positions
    pos_r[pos_r[:,0] < 0, 0] -= 0.5
    pos_r[pos_r[:,0] > 0, 0] += 0.5
    particles_r = particles_r._replace(positions = pos_r)
    
    particles_r = particles_r._replace(masses = torch.ones_like(particles_r.masses) * particles_l.masses.min() * rightState.rho * actualRatio)
    particles_l = particles_l._replace(masses = particles_l.masses * leftState.rho)
    
    particles_ = mergeParticles(particles_l, particles_r)
    
    particles = CompressibleState(
        positions = particles_.positions,
        supports = particles_.supports,
        masses = particles_.masses,
        densities = particles_.densities,
        velocities = torch.zeros_like(particles_.positions),
        
        kinds = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_.positions.shape[0], device = device, dtype = torch.int32),
        
        internalEnergies = None,
        totalEnergies = None,
        entropies = None,
        pressures = None,
        soundspeeds = None,
        divergence=torch.zeros_like(particles_.densities),
        alpha0s= torch.ones_like(particles_.densities),
        alphas= torch.ones_like(particles_.densities),
    )

    neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries

    # neighborhood, sparseNeighborhood = buildNeighborhood(particles, particles, domain, verletScale= 1.0, mode ='gather')
    # actualNeighbors = filterNeighborhood(sparseNeighborhood)

    config = {'targetNeighbors': targetNeighbors, 'domain': domain, 'support': {'iterations': 16, 'scheme': 'Monaghan'}, 'neighborhood': {'algorithm': 'compact'}}

    rho = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    # rho = computeDensity(particles, particles, domain, wrappedKernel, actualNeighbors, 'gather' )
    particles.densities = rho

    rhos = [particles.densities]
    hs = [particles.supports]

    # particles = particles._replace(positions = particles.positions + torch.randn_like(particles.positions) * particles.supports.view(-1,1) * 1e-4)

    rho, h, rhos, hs, neighborhood = evaluateOptimalSupport(particles, wrappedKernel, neighborhood, SupportScheme.Gather, config)
    particles.densities = rho
    particles.supports = h
    

    # neighborhood, sparseNeighborhood = buildNeighborhood(particles, particles, domain, verletScale= 1.0, mode ='superSymmetric')
    # neighborhood = buildSuperSymmetricNeighborhood(particles, domain, verletScale = 1.4)
    # actualNeighbors = filterNeighborhood(sparseNeighborhood)
    
    neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood)
    
    
    # omega = computeOmega(particles, particles, domain, wrappedKernel, actualNeighbors, 'gather')
    omega = computeOmega(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
    # CSR, CSC = coo_to_csrsc(actualNeighbors)


    rho = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
    particles.densities = rho
    omega = computeOmega(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
    
    rho = torch.ones_like(rho) * leftState.rho
    rho[particles.positions[:,0].abs() > 0.5] = rightState.rho
    if not smoothIC:
        rho = particles.densities

    P_initial = torch.ones_like(rho) * leftState.p
    P_initial[particles.positions[:,0].abs() > 0.5] = rightState.p
    
    mask = torch.logical_and(particles.densities > rightState.rho + 1e-3, particles.densities < leftState.rho - 1e-3)
    mask = torch.logical_or(mask, torch.logical_and(particles.densities > leftState.rho + 1e-3, particles.densities < rightState.rho - 1e-3))
    ratio = (particles.densities - rightState.rho) / (leftState.rho - rightState.rho) if rightState.rho < leftState.rho else (particles.densities - leftState.rho) / (rightState.rho - leftState.rho)
    # P_initial[mask] = leftState.p * (1 - ratio[mask]) + rightState.p * ratio[mask] if rightState.p < leftState.p else leftState.p * (1 - ratio[mask]) + rightState.p * ratio[mask]

    # P_initial = sph_op(particles, particles, domain, wrappedKernel, actualNeighbors, 'gather', operation = 'interpolate', quantity=(P_initial, P_initial))

    # print(P_initial.min(), P_initial.max(), P_initial.mean())

    u = 1 / (gamma - 1) * (P_initial / rho)
    if smoothIC:
        dx = particles_l.positions[1,0] - particles_l.positions[0,0]
        x = torch.where(particles.positions[:,0] > 0., particles.positions[:,0] - 0.5, particles.positions[:,0] + 0.5)
        ramp = torch.exp(x/dx) / (1 + torch.exp(x/dx))
        # ramp =  / (torch.exp(x/dx) + 1)
        ramped = lambda a, b, x: (a - b) / (torch.exp(x/dx) + 1) + b



        # u_max = u.max()
        # u_min = u.min()
        # u[mask] = u_min * (1 - ratio[mask]) + u_max * ratio[mask] if u_max < u_min else u_min * (1 - ratio[mask]) + u_max * ratio[mask]
        left_u = 1 / (gamma - 1) * (leftState.p / leftState.rho)
        right_u = 1 / (gamma - 1) * (rightState.p / rightState.rho)
        u = torch.where(particles.positions[:,0] > 0., ramped(left_u, right_u, x), u)
        
        # particles = particles._replace(masses = torch.where(particles.positions[:,0] > 0., ramped(leftState.rho, rightState.rho, x) * particles.masses.max(), particles.masses))
        # particles = particles._replace(masses = torch.where(particles.positions[:,0].abs() >= 0.5, particles.masses * ratio, particles.masses))
        rho = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        particles.densities = rho



        # print(ramp)

    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = rho, gamma = gamma)
    
    # P_[mask] = 0.5
    
    v_initial = torch.ones_like(rho) * leftState.v
    v_initial[particles.positions[:,0].abs() > 0.5] = rightState.v
    v_initial = v_initial.view(-1,1)
    
    neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood)
    
    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(v_initial, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particles.masses
    
    simulationState = CompressibleState(
        positions = particles.positions,
        supports = h,
        masses = particles.masses,
        densities = particles.densities,        
        velocities = v_initial,
        
        internalEnergies = u_,
        totalEnergies = totalEnergy,
        entropies = A_,
        pressures = P_,
        soundspeeds = c_s,
    
        kinds = particles.kinds,
        materials = particles.materials,
        UIDs = particles.UIDs,
    
        alphas = torch.ones_like(particles.densities),
        alpha0s = torch.ones_like(particles.densities),
        divergence=torch.zeros_like(particles.densities),
    )
    
    A_, u_, P_, c_ = idealGasEOS(A=None, P = None, u = internalEnergy, rho = simulationState.densities, gamma = gamma)
        
    # psphState = SolverSystem(
    #     positions=simulationState.positions,
    #     supports=simulationState.supports,
    #     masses=simulationState.masses,
    #     densities=simulationState.densities,
    #     velocities=simulationState.velocities,
    #     # soundspeeds=simulationState.soundspeeds,
        
    #     internalEnergies=internalEnergy,
    #     totalEnergies=totalEnergy,
    #     entropies = A_,
    #     pressures=P_,
    #     soundspeeds=c_,
        
    # )
    neighborhood, sparseNeighborhood = buildNeighborhood(simulationState, simulationState, domain, 1.4, 'superSymmetric')
    actualNeighbors = filterNeighborhood(sparseNeighborhood)

    psphSystem = SimulationSystem(
        systemState = simulationState,
        domain = domain,
        neighborhoodInfo = neighborhood,
        t = 0
    )
    
    return psphSystem
    
    

# particleSystem, domain = generateSod1D(nx, 1, initialStateLeft, initialStateRight, gamma, wrappedKernel, targetNeighbors, dtype, device, smoothIC = True)


def buildSod_reference(kernel, targetNeighbors, reference : str, nx_ = -1, dtype = torch.float32, device = torch.device('cpu'), SimulationSystem = CompressibleSystem):
    if reference == 'Price2007':
        nx = 1600 if nx_ == -1 else nx_
        gamma = 5/3
        initialStateLeft = sodInitialState(1, 1, 0)
        initialStateRight = sodInitialState(0.1, 0.125, 0)
        ratio = 8
        smoothIC = False
        timeLimit = 0.2
    elif reference == 'Monaghan2005':
        nx = 1600 if nx_ == -1 else nx_
        gamma = 1.4
        initialStateLeft = sodInitialState(1, 1, 0)
        initialStateRight = sodInitialState(0.1, 0.125, 0)
        ratio = 8
        smoothIC = True
        timeLimit = 0.2
    elif reference == 'CompSPH':
        nx = 800 if nx_ == -1 else nx_
        gamma = 5/3
        initialStateLeft = sodInitialState(1, 1, 0)
        initialStateRight = sodInitialState(0.1795, 0.25, 0)
        ratio = 1
        smoothIC = True
        timeLimit = 0.15
    else:
        nx = 1600 if nx_ == -1 else nx_
        gamma = 5/3
        initialStateLeft = sodInitialState(1, 1, 0)
        initialStateRight = sodInitialState(0.1, 0.125, 0)
        ratio = 1
        smoothIC = False
        timeLimit = 0.2


    particleSystem = generateSod1D(nx, ratio, initialStateLeft, initialStateRight, gamma, kernel, targetNeighbors, dtype, device, smoothIC = smoothIC, SimulationSystem = SimulationSystem)

    return particleSystem, gamma, timeLimit, initialStateLeft, initialStateRight
