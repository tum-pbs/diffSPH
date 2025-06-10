from typing import Union, Tuple, NamedTuple
from sphMath.integrationSchemes.util import updateStateEuler, updateStateSemiImplicitEuler
import torch
import copy
import numpy as np
from sphMath.integrationSchemes.util import split_return, preprocessSystem, postprocessSystem, finalizeSystem
from torch.profiler import record_function


# TVD RK3 scheme, uses the following integration scheme:
# $y^{n + \frac{1}{3}} = y^n + \Delta t F( y^n )$
# $y^{n + \frac{2}{3}} = \frac{3}{4}y^n + \frac{1}{4}(y^{n + \frac{1}{3}} + \Delta t F(y^{n + \frac{1}{3}}))$
# $y^{n + 1} = \frac{1}{3}y^n + \frac{2}{3}(y^{n + \frac{2}{3}}+ \Delta t F(y^{n + \frac{2}{3}}))$        
def TVDRK3(state, dt, f, *args, **kwargs):
    with record_function("[Integration] TVD RK3"):
        with record_function("[Integration] TVD RK3: k0"):
            preprocessSystem(state, dt, *args, **kwargs)
            k0, r0 = split_return(f(state, dt, *args, **kwargs))
            postprocessSystem(state, dt, r0, *args, **kwargs)
            y_1_3 = updateStateEuler(state, k0, dt)
        with record_function("[Integration] TVD RK3: k1"):
            preprocessSystem(y_1_3, 1/3 * dt, *args, **kwargs)
            k_1_3, r_1_3 = split_return(f(y_1_3, 1/3 * dt, *args, **kwargs))
            postprocessSystem(y_1_3, 1/3 * dt, r_1_3, *args, **kwargs)
            
            y_2_3 = copy.deepcopy(state)
            y_2_3.integrate(k_1_3, 1/4 * dt, selfScale = 3/4, referenceState = y_1_3, referenceWeight = 1/4)
            y_2_3.t = state.t + dt * 1/3
    
    # y_2_3 = state._replace(
    #     position = 3/4 * state.position + 1/4 * (y_1_3.position + dt * k_1_3.position),
    #     velocity = 3/4 * state.velocity + 1/4 * (y_1_3.velocity + dt * k_1_3.velocity),
    #     energy   = 3/4 * state.energy   + 1/4 * (y_1_3.energy + dt * k_1_3.energy) if hasattr(k_1_3, 'energy') else state.energy,
    #     t = state.t + dt * 2/3
    # )
        with record_function("[Integration] TVD RK3: k2"):
            preprocessSystem(y_2_3, 1/3 * dt, *args, **kwargs)
            k_2_3, r_2_3 = split_return(f(y_2_3, 1/3 * dt, *args, **kwargs))
            postprocessSystem(y_2_3, 1/3 * dt, r_2_3, *args, **kwargs)
    
        with record_function("[Integration] TVD RK3: Update"):
            finalState = copy.deepcopy(state)
            finalState.integrate(k_2_3, 2/3 * dt, selfScale = 1/3, referenceState = y_2_3, referenceWeight = 2/3)
            # finalPosition = 1/3 * state.position + 2/3 * (y_2_3.position + dt * k_2_3.position)
            # finalVelocity = 1/3 * state.velocity + 2/3 * (y_2_3.velocity + dt * k_2_3.velocity)
            # finalEnergy   = 1/3 * state.energy   + 2/3 * (y_2_3.energy + dt * k_2_3.energy) if hasattr(k_2_3, 'energy') else state.energy
            finalState.t = state.t + dt
            rs = [r0, r_1_3, r_2_3]
            ks = [k0, k_1_3, k_2_3]
            finalizeSystem(finalState, dt, rs, ks, *args, **kwargs)
    if any([t is not None for t in rs]):
        return finalState, rs
    return finalState
    
    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )
    
def TVDRK2(state, dt, f, *args, **kwargs):
    with record_function("[Integration] TVD RK2"):
        with record_function("[Integration] TVD RK2: k0"):
            preprocessSystem(state, dt, *args, **kwargs)
            k0, r0 = split_return(f(state, dt, *args, **kwargs))
            postprocessSystem(state, dt, r0, *args, **kwargs)
            
            state1 = updateStateEuler(state, k0, dt)
        with record_function("[Integration] TVD RK2: k1"):
            preprocessSystem(state1, 1/2 * dt, *args, **kwargs)
            k1, r1 = split_return(f(state1, 1/2 * dt, *args, **kwargs))
            postprocessSystem(state1, 1/2 * dt, r1, *args, **kwargs)
        
        with record_function("[Integration] TVD RK2: Update"):
            finalState = copy.deepcopy(state)
            finalState.integrate(k1, 1/2 * dt, selfScale = 1/2, referenceState = state1, referenceWeight = 1/2)
            finalState.t = state.t + dt
            
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, dt, rs, ks, *args, **kwargs)
            if any([t is not None for t in rs]):
                return finalState, rs
        return finalState
    
    # finalPosition = 1/2 * state.position + 1/2 * (state1.position + dt * k1.position)
    # finalVelocity = 1/2 * state.velocity + 1/2 * (state1.velocity + dt * k1.velocity)
    # finalEnergy   = 1/2 * state.energy   + 1/2 * (state1.energy + dt * k1.energy) if hasattr(k1, 'energy') else state.energy
    
    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )
    