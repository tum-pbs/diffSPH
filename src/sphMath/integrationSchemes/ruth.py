from sphMath.integrationSchemes.util import updateStateEuler, updateStateSemiImplicitEuler
import torch
import copy
from sphMath.integrationSchemes.util import split_return, preprocessSystem, postprocessSystem, finalizeSystem
from torch.profiler import record_function

    
# Based on PySPH based on
# [Omeylan2002] I.M. Omelyan, I.M. Mryglod and R. Folk, "Optimized
# Forest-Ruth- and Suzuki-like algorithms for integration of motion
# in many-body systems", Computer Physics Communications 146, 188 (2002)
# http://arxiv.org/abs/cond-mat/0110585
# Probably won't work well with energy or other integration terms due to relying on an initial drift step
def PEFRL(state, dt, f, *args, **kwargs):
    with record_function("[Integration] PEFRL"):
        lamda = -0.2123418310626054
        xi = +0.1786178958448091
        chi = -0.06626458266981849
        with record_function("[Integration] PEFRL: Step 1"):
            state1 = copy.deepcopy(state)
            state1.integratePosition([], [], semiImplicitScale = xi * dt)
            state1.t = state.t + xi * dt
            preprocessSystem(state1, xi * dt, *args, **kwargs)
            k0, r0 = split_return(f(state1, xi * dt, *args, **kwargs))
            postprocessSystem(state1, xi * dt, r0, *args, **kwargs)
            state1.integrateVelocity(k0, (1 - 2 * lamda) * dt / 2)
            state1.integrateQuantities(k0, (1 - 2 * lamda) * dt / 2)
        
        
        # r1 = state.position + xi * dt * state.velocity
        # k0 = f(state._replace(position = r1, t = state.t + xi * dt))
        # v1 = state.velocity + (1 - 2 * lamda) * dt / 2 * k0.velocity
        # e1 = state.energy + (1 - 2 * lamda) * dt / 2 * k0.energy if hasattr(k0, 'energy') else state.energy
        # state1 = state._replace(
        #     position = r1,
        #     velocity = v1,
        #     energy = e1,
        #     t = state.t + xi * dt
        # )
        with record_function("[Integration] PEFRL: Step 2"):
            state2 = copy.deepcopy(state1)
            state2.integratePosition([], [], semiImplicitScale = chi * dt)
            state2.t = state1.t + chi * dt
            preprocessSystem(state2, chi * dt, *args, **kwargs)
            k1, r1 = split_return(f(state2, chi * dt, *args, **kwargs))
            postprocessSystem(state2, chi * dt, r1, *args, **kwargs)
            state2.integrateVelocity(k1, lamda * dt)
            state2.integrateQuantities(k1, lamda * dt)
        
        # r2 = state1.position + chi * dt * state1.velocity
        # k1 = f(state1._replace(position = r2, t = state1.t + chi * dt))
        # v2 = state1.velocity + lamda * dt * k1.velocity
        # e2 = state1.energy + lamda * dt * k1.energy if hasattr(k1, 'energy') else state1.energy
        # state2 = state1._replace(
        #     position = r2,
        #     velocity = v2,
        #     energy = e2,
        #     t = state1.t + chi * dt
        # )
        with record_function("[Integration] PEFRL: Step 3"):
            state3 = copy.deepcopy(state2)
            state3.integratePosition([], [], semiImplicitScale = (1 - 2 * (chi + xi)) * dt)
            state3.t = state2.t + (1 - 2 * (chi + xi)) * dt
            preprocessSystem(state3, (1 - 2 * (chi + xi)) * dt, *args, **kwargs)
            k2, r2 = split_return(f(state3, (1 - 2 * (chi + xi)) * dt, *args, **kwargs))
            postprocessSystem(state3, (1 - 2 * (chi + xi)) * dt, r2, *args, **kwargs)
            state3.integrateVelocity(k2, lamda * dt)
            state3.integrateQuantities(k2, lamda * dt)
        
        
        # r3 = r2 + (1 - 2 * (chi + xi)) * dt * v2
        # k2 = f(state2._replace(position = r3, t = state2.t + (1 - 2 * (chi + xi)) * dt))
        # v3 = v2 + lamda * dt * k2.velocity
        # e3 = e2 + lamda * dt * k2.energy if hasattr(k2, 'energy') else e2
        # state3 = state2._replace(
        #     position = r3,
        #     velocity = v3,
        #     energy = e3,
        #     t = state2.t + (1 - 2 * (chi + xi)) * dt
        # )
        with record_function("[Integration] PEFRL: Step 4"):
            finalState = copy.deepcopy(state3)
            finalState.integratePosition([], [], semiImplicitScale = chi * dt)
            finalState.t = state3.t + chi * dt
            preprocessSystem(finalState, chi * dt, *args, **kwargs)
            k3, r3 = split_return(f(finalState, chi * dt, *args, **kwargs))
            postprocessSystem(finalState, chi * dt, r3, *args, **kwargs)
        
        with record_function("[Integration] PEFRL: Update"):
            finalState.integrateVelocity(k3, (1 - 2 * lamda) * dt / 2)
            finalState.integrateQuantities(k3, (1 - 2 * lamda) * dt / 2)
            finalState.integratePosition([], [], semiImplicitScale = xi * dt)
            finalState.t = state.t + dt
        
            rs = [r0, r1, r2, r3]
            ks = [k0, k1, k2, k3]
            finalizeSystem(finalState, dt, rs, ks, *args, **kwargs)
        if any([t is not None for t in rs]):
            return finalState, rs
        return finalState
    
    # r4 = r3 + chi * dt * v3
    # k3 = f(state3._replace(position = r4, t = state3.t + chi * dt))
    
    # finalVelocity = v3 + (1 - 2 * lamda) * dt / 2 * k3.velocity
    # finalPosition = r4 + xi * dt * finalVelocity
    # finalEnergy = e3 + (1 - 2 * lamda) * dt / 2 * k3.energy if hasattr(k3, 'energy') else e3
    
    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )
def VEFRL(state, dt, f, *args, **kwargs):
    with record_function("[Integration] VEFRL"):
        lamda = -0.2123418310626054
        xi = +0.1786178958448091
        chi = -0.06626458266981849
        with record_function("[Integration] VEFRL: Step 1"):
            preprocessSystem(state, xi * dt, *args, **kwargs)
            k0, r0 = split_return(f(state, xi * dt, *args, **kwargs))
            postprocessSystem(state, r0, *args, **kwargs)
            state1 = copy.deepcopy(state)
            state1.integrateVelocity(k0, xi * dt)
            state1.integrateQuantities(k0, xi * dt)
            state1.integratePosition(k0, 0.0, semiImplicitScale =  (1 - 2 * lamda) * dt / 2)
        
        # v1 = state.velocity + xi * dt * k0.velocity
        # r1 = state.position + (1 - 2 * lamda) * dt / 2 * v1
        # e1 = state.energy + xi * dt * k0.energy if hasattr(k0, 'energy') else state.energy
        # state1 = state._replace(
        #     position = r1,
        #     velocity = v1,
        #     energy = e1,
        #     t = state.t + xi * dt
        # )
        with record_function("[Integration] VEFRL: Step 2"):
            preprocessSystem(state1, chi * dt, *args, **kwargs)
            k1, r1 = split_return(f(state1, chi * dt, *args, **kwargs))
            postprocessSystem(state1, r1, *args, **kwargs)
            state2 = copy.deepcopy(state1)
            state2.integrateVelocity(k1, chi * dt)
            state2.integrateQuantities(k1, chi * dt)
            state2.integratePosition(k0, 0.0, semiImplicitScale = lamda * dt)
        
        # v2 = state1.velocity + chi * dt * k1.velocity
        # r2 = state1.position + lamda * dt * v2
        # e2 = state1.energy + chi * dt * k1.energy if hasattr(k1, 'energy') else state1.energy
        # state2 = state1._replace(
        #     position = r2,
        #     velocity = v2,
        #     energy = e2,
        #     t = state1.t + chi * dt
        # )
        with record_function("[Integration] VEFRL: Step 3"):
            preprocessSystem(state2, (1 - 2 * (chi + xi)) * dt, *args, **kwargs)
            k2,r2 = split_return(f(state2, (1 - 2 * (chi + xi)) * dt, *args, **kwargs))
            postprocessSystem(state2, r2, *args, **kwargs)
            state3 = copy.deepcopy(state2)
            state3.integrateVelocity(k2, (1 - 2 * (chi + xi)) * dt)
            state3.integrateQuantities(k2, (1 - 2 * (chi + xi)) * dt)
            state3.integratePosition(k0, 0.0, semiImplicitScale = lamda * dt)
        
        
        # v3 = state2.velocity + (1 - 2 * (chi + xi)) * dt * k2.velocity
        # r3 = state2.position + lamda * dt * v3
        # e3 = state2.energy + (1 - 2 * (chi + xi)) * dt * k2.energy if hasattr(k2, 'energy') else state2.energy
        # state3 = state2._replace(
        #     position = r3,
        #     velocity = v3,
        #     energy = e3,
        #     t = state2.t + (1 - 2 * (chi + xi)) * dt
        # )
        with record_function("[Integration] VEFRL: Step 4"):
            preprocessSystem(state3, chi * dt, *args, **kwargs)
            k3, r3 = split_return(f(state3, chi * dt, *args, **kwargs))
            postprocessSystem(state3, r3, *args, **kwargs)  
            state4 = copy.deepcopy(state3)
            state4.integrateVelocity(k3, chi * dt)
            state4.integrateQuantities(k3, chi * dt)
            state4.integratePosition(k0, 0.0, semiImplicitScale = (1 - 2 * lamda) * dt / 2)
        
        # v4 = state3.velocity + chi * dt * k3.velocity
        # r4 = state3.position + (1 - 2 * lamda) * dt / 2 * v4
        # e4 = state3.energy + chi * dt * k3.energy if hasattr(k3, 'energy') else state3.energy
        # state4 = state3._replace(
        #     position = r4,
        #     velocity = v4,
        #     energy = e4,
        #     t = state3.t + chi * dt
        # )
        with record_function("[Integration] VEFRL: Step5"):
            preprocessSystem(state4, xi * dt, *args, **kwargs)
            k4, r4 = split_return(f(state4, xi * dt, *args, **kwargs))
            postprocessSystem(state4, r4, *args, **kwargs)
        with record_function("[Integration] VEFRL: Update"):
            finalState = copy.deepcopy(state4)
            finalState.integrateVelocity(k4, xi * dt)
            finalState.integrateQuantities(k4, xi * dt)
            finalState.t = state.t + dt
            
            rs = [r0, r1, r2, r3, r4]
            ks = [k0, k1, k2, k3, k4]
            finalizeSystem(finalState, dt, rs, ks, *args, **kwargs)
            if any([t is not None for t in rs]):
                return finalState, rs

        return finalState
    
    
    # finalVelocity = v4 + xi * dt * k4.velocity
    # finalPosition = r4
    # finalEnergy = e4 + xi * dt * k4.energy if hasattr(k4, 'energy') else e4
    
    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )