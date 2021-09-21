# pylint: disable=no-member, invalid-name, line-too-long
"""
ODE flow
"""
import itertools


def flow(x_0, prepare, make_step, compare, dt_amplification=1.1, dt_damping=10.0, checkpoints=None):
    """sample the dt_i to obtain a smooth  { x(t_i) }_i

    Parameters
    ----------
    x_0 : Any
        initial state

    prepare : function
        function that takes ``(current_state, current_time, previous_data, previous_time)`` and returns ``current_data``

    make_step : function
        function that takes ``(current_state, current_data, current_time, dt)`` and returns ``next_state``

    compare : function
        function that takes ``(previous_data, current_data)`` and returns a float or a list of floats
    """
    dt = 1
    last_dt = 0
    step_change_dt = 0
    t = 0
    d = None
    next_t = 0
    if checkpoints is not None:
        checkpoints = iter(checkpoints)
        try:
            next_t = next(checkpoints)
        except StopIteration:
            return

    data = prepare(x_0, t, None, 0)
    x = x_0

    for step in itertools.count():

        state = {
            'step': step,
            't': t,
            'dt': last_dt,
            'd': d,
        }
        internals = {
            'x': x,
            'data': data,
            'changed_dt': step_change_dt == step - 1,
        }

        if t >= next_t:
            yield state, internals

            if checkpoints is not None:
                try:
                    next_t = next(checkpoints)
                except StopIteration:
                    return

        while True:
            # 2 - Make a tentative step
            if t + dt < next_t:
                new_x = make_step(x, data, t, dt)
                new_t = t + dt
                last_dt = dt
            else:
                new_t = next_t
                last_dt = next_t - t
                new_x = make_step(x, data, t, last_dt)

            # 3 - Check if the step is small enough
            new_data = prepare(new_x, new_t, data, t)

            d = compare(data, new_data)
            if not isinstance(d, tuple):
                d = (d,)
            if all(c < 1 for c in d):
                if all(c < 1/2 for c in d):
                    dt *= dt_amplification
                break

            # 4 - If not, reset and retry
            dt /= dt_damping
            step_change_dt = step

        # 5 - If yes, compute the new output and gradient
        x = new_x
        t = new_t
        data = prepare(x, t, new_data, new_t)
