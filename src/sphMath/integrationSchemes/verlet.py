from typing import Union, Tuple, NamedTuple
from sphMath.integrationSchemes.util import updateStateEuler, updateStateSemiImplicitEuler
import torch
import copy
import numpy as np
from sphMath.integrationSchemes.util import split_return, preprocessSystem, postprocessSystem, finalizeSystem, updateStep, initializeSystem
from torch.profiler import record_function


# Also known as synchronized form
def leapFrog(initialState, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(initialState, dt, *args, **kwargs)

    with record_function("[Integration] Leap Frog"):
        with record_function("[Integration] Leap Frog: k0"):
            currentState = initialState.initializeNewState(*args, **kwargs)
            k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs) if priorStep is None else priorStep
            # k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs)

            halfState = initialState.initializeNewState(*args, **kwargs)
            halfState.integratePosition(k0, dt, verletScale=0.5 * dt**2)
            halfState.t = initialState.t + 0.5 * dt

        with record_function("[Integration] Leap Frog: k1"):
            k1, r1 = updateStep(initialState, halfState, dt / 2, f, *args, **kwargs)
    
        with record_function("[Integration] Leap Frog: Update"):
            finalState = initialState.initializeNewState(*args, **kwargs)
            finalState.integratePosition(k0, dt, verletScale=0.5 * dt**2)
            finalState.integrateVelocity([k0, k1], [0.5 * dt, 0.5 * dt])
            finalState.integrateQuantities([k0, k1], [0.5 * dt, 0.5 * dt])
            finalState.t = initialState.t + dt
            
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, dt, rs, ks, [0.5, 0.5], *args, **kwargs)
            if any([t is not None for t in rs]):
                return finalState, rs, ks
    return finalState, ks

# Also known as kick-drift-kick form and position verlet
# see 'Improvements in SPH method by means of interparticle
# contact algorithm and analysis of perforation tests at moderate
# projectile velocities.'
def symplecticEuler(initialState, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(initialState, dt, *args, **kwargs)
    with record_function("[Integration] Symplectic Euler"):
        with record_function("[Integration] Symplectic Euler: k0"):
            currentState = initialState.initializeNewState(*args, **kwargs)
            k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs) if priorStep is None else priorStep
            # k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs)
            halfState = initialState.initializeNewState(*args, **kwargs)
            halfState.integrate(k0, dt / 2, **kwargs)

        with record_function("[Integration] Symplectic Euler: k1"):
            k1, r1 = updateStep(initialState, halfState, dt / 2, f, *args, **kwargs)

        with record_function("[Integration] Symplectic Euler: Update"):
            finalState = initialState.initializeNewState(*args, **kwargs)
            finalState.integrateVelocity(k1, dt, **kwargs)
            finalState.integratePosition(k0, dt / 2, semiImplicitScale = dt / 2, **kwargs)
            if hasattr(finalState, 'integrateDensity'):
                finalState.integrateDensity(k1, dt, **kwargs)
                finalState.integrateQuantities(k1, dt, densitySwitch = True, **kwargs)
            else:
                finalState.integrateQuantities(k1, dt, **kwargs)
            finalState.t = initialState.t + dt
            
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, initialState, dt, rs, ks, [0,1], *args, **kwargs)
            if any([t is not None for t in rs]):
                return finalState, rs, ks
    return finalState, ks

    # finalVelocity = state.velocity + dt * k1.velocity
    # finalPosition = state.position + dt * (k0.position + finalVelocity) / 2
    # finalEnergy   = state.energy + dt * k1.energy if hasattr(k1, 'energy') else state.energy

    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )

# velocity verlet, doesn't work well for systems with energy, also known as drift-kick-drift
def velocityVerlet(initialState, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(initialState, dt, *args, **kwargs)
    with record_function("[Integration] Velocity Verlet"):
        with record_function("[Integration] Velocity Verlet: k0"):
            currentState = initialState.initializeNewState(*args, **kwargs)
            k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs) if priorStep is None else priorStep
            # k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs)
            halfState = initialState.initializeNewState(*args, **kwargs)
            halfState.integrateVelocity(k0, dt / 2)
            halfState.integratePosition(k0, 0.0, semiImplicitScale = dt)
            halfState.t = initialState.t + dt / 2

        with record_function("[Integration] Velocity Verlet: k1"):
            k1, r1 = updateStep(initialState, halfState, dt / 2, f, *args, **kwargs)

        with record_function("[Integration] Velocity Verlet: Update"):
            finalState = halfState.initializeNewState(*args, **kwargs)
            finalState.integrateVelocity(k1, dt / 2)
            finalState.integrateQuantities(k1, dt)
            
            finalState.t = initialState.t + dt
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, initialState, dt, rs, ks, [1/2, 1/2], *args, **kwargs)
            if any([t is not None for t in rs]):
                return finalState, rs, ks
    return finalState, ks
    
    k0 = f(state)
    halfStateVelocity = state.velocity + k0.velocity * dt / 2
    finalPosition = state.position + halfStateVelocity * dt
    halfState = state._replace(position = finalPosition, velocity = halfStateVelocity)

    k1 = f(halfState)
    finalVelocity = halfStateVelocity + k1.velocity * dt / 2
    finalEnergy   = state.energy + dt * k0.energy if hasattr(k0, 'energy') else state.energy

    return state._replace(
        position = finalPosition,
        velocity = finalVelocity,
        energy = finalEnergy,
        t = state.t + dt
    )

