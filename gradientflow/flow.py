# pylint: disable=no-member, invalid-name, line-too-long
"""
ODE flow
"""
import itertools


def flow(x, prepare, make_step, compare, dt_amplification=1.1, dt_damping=10.0):
    """
    sample the dt_i to obtain a smooth  { x(t_i) }_i
    """
    dt = 1
    last_dt = 0
    step_change_dt = 0
    t = 0
    d = None

    data = prepare(x, t, None, 0)

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

        yield state, internals

        while True:
            # 2 - Make a tentative step
            new_x = make_step(x, data, t, dt)
            new_t = t + dt
            last_dt = dt

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
