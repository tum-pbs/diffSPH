from typing import Union, Tuple, NamedTuple
from sphMath.integrationSchemes.util import updateStateEuler, updateStateSemiImplicitEuler
import torch
import copy
import numpy as np
from sphMath.integrationSchemes.util import split_return, preprocessSystem, postprocessSystem, finalizeSystem, updateStep, initializeSystem
from torch.profiler import record_function


class butcherTableau(NamedTuple):
    a: np.array
    b: Union[np.array,Tuple[np.array, np.array]]
    c: np.array


def RungeKuttaB(initialState, dt, f, butcherTableau, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(initialState, dt, *args, **kwargs)
    with record_function("[Integration] Butcher"):
        currentState = initialState.initializeNewState(*args, **kwargs)
        k0, r0 = updateStep(initialState, currentState, dt, f, *args, **kwargs) if priorStep is None else priorStep
        ks = [k0]
        rs = [r0]
        for ic, c in enumerate(butcherTableau.c[1:]):
            with record_function(f"[Integration] Butcher: k{ic+1} prep"):
                current_as = butcherTableau.a[ic+1,:ic+1]
                currentState = initialState.initializeNewState(*args, **kwargs)
                for i, a in enumerate(current_as):
                    if a != 0:
                        currentState = updateStateEuler(currentState, ks[i], a * dt, copyState = False, **kwargs)
                currentState.t = initialState.t + c * dt
            with record_function(f"[Integration] Butcher: k{ic+1}"):
                k, r = updateStep(initialState, currentState, dt, f, *args, **kwargs)
                ks.append(k)
                rs.append(r)
        
        with record_function("[Integration] Butcher: Update"):
            if not isinstance(butcherTableau.b, tuple):
                new_state = initialState.initializeNewState(*args, **kwargs)
                for i, b in enumerate(butcherTableau.b):
                    if b != 0:
                        new_state = updateStateEuler(new_state, ks[i], b * dt, **kwargs)
                new_state.t = initialState.t + dt
                finalizeSystem(new_state, initialState, dt, rs, ks, butcherTableau.b, *args, **kwargs)
                if any([t is not None for t in rs]):
                    return new_state, rs, ks
                return new_state, ks
            else:
                new_states = []
                for b_ in butcherTableau.b:
                    new_state = updateStateEuler(new_state, ks[i], b * dt, **kwargs)
                    for i, b in enumerate(b_):
                        if b != 0:
                            new_state = updateStateEuler(new_state, ks[i], b * dt, **kwargs)
                    new_state.t = initialState.t + dt
                    new_states.append(new_state)
                for new_state in new_states:
                    finalizeSystem(new_state, initialState, dt, rs, ks, b_, *args, **kwargs)
                if any([t is not None for t in rs]):
                    return new_states, rs, ks
                else:
                    return new_states, ks


def getButcherTableau(scheme, alpha = 1/2, beta = 2/3):
    if scheme == 'forwardEuler':
        return butcherTableau(
            a = np.array([[0]]),
            b = np.array([1]),
            c = np.array([0])
        )
    elif scheme == 'generic2nd':
        return butcherTableau(
            a = np.array([[0, 0], [alpha, 0]]),
            b = np.array([1 - 1/(2*alpha), 1/(2*alpha)]),
            c = np.array([0, alpha])
        )
    elif scheme == 'midpoint':
        return butcherTableau(
            a = np.array([[0, 0], [1/2, 0]]),
            b = np.array([0, 1]),
            c = np.array([0, 1/2])
        )
    elif scheme == 'heunsMethod':
        return butcherTableau(
            a = np.array([[0, 0], [1, 0]]),
            b = np.array([1/2, 1/2]),
            c = np.array([0, 1])
        )
    elif scheme == 'ralston':
        return butcherTableau(
            a = np.array([[0, 0], [2/3, 0]]),
            b = np.array([1/4, 3/4]),
            c = np.array([0, 2/3])
        )
    elif scheme == 'generic3rd':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [alpha, 0, 0],
                          [beta / alpha * (beta - 3 * alpha * (1-alpha)) / (3 * alpha - 2), - beta/alpha *(beta - alpha) / (3 *alpha - 2), 0]]),
            b = np.array([1 - (3 * alpha + 3 * beta -2) / ( 6 * alpha * beta), (3 * beta - 2) / (6 * alpha * (beta - alpha)), (2 -3 *alpha / (6 * beta * (beta - alpha)))]),
            c = np.array([0, alpha, beta])
        )
    elif scheme == 'RK3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [1/2, 0, 0],
                          [-1, 2, 0]]),
            b = np.array([1/6, 2/3, 1/6]),
            c = np.array([0, 1/2, 1])
        )
    elif scheme == 'Heun3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [1/3, 0, 0],
                          [0, 2/3, 0]]),
            b = np.array([1/4, 0, 3/4]),
            c = np.array([0, 1/3, 2/3])
        )
    elif scheme == 'ralston3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [1/2, 0, 0],
                          [0, 3/4, 0]]),
            b = np.array([2/9, 1/3, 4/9]),
            c = np.array([0, 1/2, 3/4])
        )
    elif scheme == 'Wray3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [8/15, 0, 0],
                          [1/4, 5/12, 0]]),
            b = np.array([1/4, 0, 3/4]),
            c = np.array([0, 8/15, 2/3])
        )
    elif scheme == 'SSPRK3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [1, 0, 0],
                          [1/4, 1/4, 0]]),
            b = np.array([1/6, 1/6, 2/3]),
            c = np.array([0, 1, 1/2])
        )
    elif scheme == 'RK4':
        return butcherTableau(
            a = np.array([[0, 0, 0, 0], 
                          [1/2, 0, 0, 0],
                          [0, 1/2, 0, 0],
                          [0, 0, 1, 0]]),
            b = np.array([1/6, 1/3, 1/3, 1/6]),
            c = np.array([0, 1/2, 1/2, 1])
        )
    elif scheme == 'RK4alt':
        return butcherTableau(
            a = np.array([[0, 0, 0, 0], 
                          [1/3, 0, 0, 0],
                          [-1/3, 1, 0, 0],
                          [1, -1, 1, 0]]),
            b = np.array([1/8, 3/8, 3/8, 1/8]),
            c = np.array([0, 1/3, 2/3, 1])
        )
    elif scheme == 'Nystrom5':
        return butcherTableau(
            a = np.array([[0,       0,      0,     0, 0,0], 
                          [1/3,     0,      0,     0, 0, 0],
                          [4/25, 6/25,      0,     0, 0, 0],
                          [1/4,     -3,  15/4,     0, 0, 0],
                          [2/27,  10/9, -50/81, 8/81, 0, 0],
                          [2/25, 12/25,   2/15, 8/75, 0 ,0]]),
            b = np.array([23/192, 0, 125/192,  0, -27/64, 125/192]),
            c = np.array([0, 1/3, 2/5, 1, 2/3, 4/5])
        )
    else:
        raise ValueError(f"Unknown scheme {scheme}")
    

# Evaluate-Predict-Evaluate-Correct (EPEC) scheme based on pySPH code
# is equivalent to the traditional explicit midpoint method 
def EPEC(state, dt, f, *args, **kwargs):
    return RungeKuttaB(state, dt, f, getButcherTableau('midpoint'), *args, **kwargs)
# Modified EPEC scheme based on pySPH code, uses $y^{n+1} = y^n + \frac{\Delta t}{2}\left( F(y^n) + F(y^{n+\frac{1}{2}}) \right)$
def EPECmodified(state, dt, f, *args, **kwargs):
    return RungeKuttaB(state, dt, f, getButcherTableau('heunsMethod'), *args, **kwargs)
    # Implementation based on PySPH code, doesn't really work
    # k0 = f(state) 
    # halfState = updateFn(state, k0, dt / 2)
    # k1 = f(halfState)

    # finalPosition = state.position + dt * (k0.position + k1.position) / 2
    # finalVelocity = state.velocity + dt * (k0.velocity + k1.velocity) / 2
    # finalEnergy   = state.energy + dt * (k0.energy + k1.energy) / 2 if hasattr(k0, 'energy') else state.energy

    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )


forwardEuler    = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('forwardEuler'), *args, **kwargs)
RungeKutta2     = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('midpoint'), *args, **kwargs)
midPoint        = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('midpoint'), *args, **kwargs)
heunsMethod     = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('heunsMethod'), *args, **kwargs)
ralston2nd      = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('ralston'), *args, **kwargs)
RungeKutta3     = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('RK3'), *args, **kwargs)
heunsMethod3rd  = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('Heun3'), *args, **kwargs)
ralston3rd      = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('ralston3'), *args, **kwargs)
Wray3rd         = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('Wray3'), *args, **kwargs)
SSPRK3          = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('SSPRK3'), *args, **kwargs)
RungeKutta4     = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('RK4'), *args, **kwargs)
RungeKutta4alt  = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('RK4alt'), *args, **kwargs)
Nystrom5th      = lambda state, dt, f, *args, **kwargs: RungeKuttaB(state, dt, f, getButcherTableau('Nystrom5'), *args, **kwargs)
