# pylint: disable=no-member, invalid-name, line-too-long
"""
Gradient flow for an ODE
"""
import copy
import itertools

import torch


def gradientflow_ode(var0, grad_fn, max_dgrad):
    """
    gradientflow for an ODE
    """
    var = copy.deepcopy(var0)

    dt = 1
    current_dt = 0

    t = 0

    grad = grad_fn(var)
    dgrad = 0

    custom_internals = None
    if isinstance(grad, tuple):
        grad, custom_internals = grad

    for step in itertools.count():

        state = {
            'step': step,
            't': t,
            'dt': current_dt,
            'dgrad': dgrad,
        }
        internals = {
            'variables': var,
            'gradient': grad,
            'custom': custom_internals,
        }

        yield state, internals

        if torch.isnan(grad).any():
            break

        # 1 - Save current state
        state = copy.deepcopy((var, t))

        while True:
            # 2 - Make a tentative step
            var.add_(dt * grad)
            t += dt
            current_dt = dt

            # 3 - Check if the step is small enough
            new_grad = grad_fn(var)
            if isinstance(new_grad, tuple):
                new_grad, custom_internals = new_grad

            if torch.isnan(new_grad).any():
                break

            if grad.norm() == 0 or new_grad.norm() == 0:
                dgrad = 0
            else:
                dgrad = (grad - new_grad).norm().pow(2).div(grad.norm() * new_grad.norm()).item()

            if dgrad < max_dgrad:
                if dgrad < 0.5 * max_dgrad:
                    dt *= 1.1
                break

            # 4 - If not, reset and retry
            dt /= 10

            var, t = copy.deepcopy(state)

        # 5 - If yes, compute the new output and gradient
        grad = new_grad
