from sphMath.integrationSchemes.util import updateStateEuler, updateStateSemiImplicitEuler
import torch
import copy
from sphMath.integrationSchemes.util import split_return, preprocessSystem, postprocessSystem, finalizeSystem, updateStep, initializeSystem
from torch.profiler import record_function
import warnings

def integrateExplicitEuler(initialState, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    if priorStep is not None:
        warnings.warn("Prior step is not used in Explicit Euler integration. Ignoring it.")
    initializeSystem(initialState, dt, *args, **kwargs)
    with record_function("[Integration] Explicit Euler"):
        with record_function("[Integration] Explicit Euler: Eval"):
            currentState = initialState.initializeNewState(*args, **kwargs)
            k1, r1 = updateStep(initialState, currentState, dt, f, *args, **kwargs)
        with record_function("[Integration] Explicit Euler: Update"):
            newState = updateStateEuler(initialState, k1, dt, **kwargs)
            finalizeSystem(newState, initialState, dt, [r1], [k1], [1], *args, **kwargs)
        return (newState, [k1] )if r1 is None else (newState, [r1], [k1])

def integrateSemiImplicitEuler(initialState, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    if priorStep is not None:
        warnings.warn("Prior step is not used in Semi-Implicit Euler integration. Ignoring it.")
    initializeSystem(initialState, dt, *args, **kwargs)
    with record_function("[Integration] Semi-Implicit Euler"):
        with record_function("[Integration] Semi-Implicit Euler: Eval"):
            currentState = initialState.initializeNewState(*args, **kwargs)
            k1, r1 = updateStep(initialState, currentState, dt, f, *args, **kwargs)
        with record_function("[Integration] Semi-Implicit Euler: Update"):
            newState = updateStateSemiImplicitEuler(initialState, k1, dt, **kwargs)
            finalizeSystem(newState, initialState, dt, [r1], [k1], [1], *args, **kwargs)
        return (newState, [k1] )if r1 is None else (newState, [r1], [k1])
