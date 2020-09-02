# pylint: disable=no-member, invalid-name, line-too-long
"""
This file implements a continuous version of momentum SGD
Dynamics that compares the angle of the gradient between steps and keep it small

- stop when margins are reached

It contains two implementation of the same dynamics:
1. `train_regular` for any kind of models
2. `train_kernel` only for linear models
"""
from .backprop import gradientflow_backprop
from .kernel import gradientflow_kernel
from .ode import gradientflow_ode


__all__ = ['gradientflow_backprop', 'gradientflow_kernel', 'gradientflow_ode']
