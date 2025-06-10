
from sphMath.operations import sph_operation
import torch
from torch.autograd import gradcheck

def try_individual(test_fn, print_exception, t, *args):
    t.requires_grad = True
    try:
        test = gradcheck(test_fn, rtol = 1e-2, *args)
    except Exception as e:
        if print_exception:
            print(e)
        test = False
    if test:
        print('✅', end = '')
    else:
        print('❌', end = '')
    t.requires_grad = False
def try_gradients(test_fn, print_exception, *args):
    try:
        test = gradcheck(test_fn, *args)
    except Exception as e:
        if print_exception:
            print(e)
        test = False
    if test:
        print('✅', end = '')
    else:
        print('❌', end = '')


def test_gradients(print_exception, mass, pos, supports, rho, q, mass2, pos2, supports2, rho2, q2, wrappedKernel, i, j, minExtent, maxExtent, periodic, **kwargs):
    mass_ = mass.to(torch.float64).detach()
    pos_ = pos.to(torch.float64).detach()
    supports_ = supports.to(torch.float64).detach()
    rho_ = rho.to(torch.float64).detach()
    q_ = q.to(torch.float64).detach()

    mass2_ = mass2.to(torch.float64).detach()
    pos2_ = pos2.to(torch.float64).detach()
    supports2_ = supports2.to(torch.float64).detach()
    rho2_ = rho2.to(torch.float64).detach()
    q2_ = q2.to(torch.float64).detach()


    test_fn = lambda mass, mass2, rho, rho2, q, q2, pos, pos2, supports, supports2: sph_operation(
        (mass, mass2), (rho, rho2), (q, q2), 
        (pos, pos2), (supports, supports2), wrappedKernel, i, j,
        minExtent, maxExtent, periodic, **kwargs)
    
    mass2_.requires_grad = False
    rho2_.requires_grad = False
    q2_.requires_grad = False
    pos2_.requires_grad = False
    supports2_.requires_grad = False

    mass_.requires_grad = False
    rho_.requires_grad = False
    q_.requires_grad = False
    pos_.requires_grad = False
    supports_.requires_grad = False

    print('Checking ', end = '')
    if 'operation' in kwargs:
        print(f'{kwargs["operation"]}', end = '')
    else:
        print('???', end = '')
    if 'gradientMode' in kwargs:
        gradMode = kwargs['gradientMode']
        gradModeStr = gradMode[:3]
        print(f'[{gradModeStr}]', end = '')
    if 'laplaceMode' in kwargs:
        lapMode = kwargs['laplaceMode']
        lapModeStr = lapMode[:3]
        print(f'[{lapModeStr}]', end = '')
    if 'divergenceMode' in kwargs:
        divMode = kwargs['divergenceMode']
        divModeStr = divMode[:3]
        print(f'[{divModeStr}]', end = '')
    if 'consistentDivergence' in kwargs:
        print(f'[{"c" if kwargs["consistentDivergence"] else " "}]', end = '')


    # for key, value in kwargs.items():
    #     print(f'{key}: {value}, ', end = '')
    print('{mrqxh}\t', end = '')
    print('[i]:\t', end = '')    
    try_individual(test_fn, print_exception, mass_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    try_individual(test_fn, print_exception, rho_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    try_individual(test_fn, print_exception, q_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    try_individual(test_fn, print_exception, pos_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    try_individual(test_fn, print_exception, supports_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))

    print('\t[j]: ', end = '')
    try_individual(test_fn, print_exception, mass2_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    try_individual(test_fn, print_exception, rho2_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    try_individual(test_fn, print_exception, q2_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    try_individual(test_fn, print_exception, pos2_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    try_individual(test_fn, print_exception, supports2_, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))

    print('\t[all]: ', end = '')
    
    mass2_.requires_grad = True
    rho2_.requires_grad = True
    q2_.requires_grad = True
    pos2_.requires_grad = True
    supports2_.requires_grad = True

    mass_.requires_grad = True
    rho_.requires_grad = True
    q_.requires_grad = True
    pos_.requires_grad = True
    supports_.requires_grad = True
    try_gradients(test_fn, True, (mass_, mass2_, rho_, rho2_, q_, q2_, pos_, pos2_, supports_, supports2_))
    print('')