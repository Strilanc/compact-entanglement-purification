from typing import Iterator, Literal

import stim

from epu.err_model import ErrModel
from epu.stabilizer_code import rep_xx, rep_yy, rep_zz


def iter_failures(m1: ErrModel, m2: ErrModel) -> Iterator[tuple[stim.PauliString, float]]:
    x1 = m1.x
    y1 = m1.y
    z1 = m1.z
    x2 = m2.x
    y2 = m2.y
    z2 = m2.z
    items1 = [
        ("I", 1 - x1 - y1 - z1),
        ("X", x1),
        ("Y", y1),
        ("Z", z1),
    ]
    items2 = [
        ("I", 1 - x2 - y2 - z2),
        ("X", x2),
        ("Y", y2),
        ("Z", z2),
    ]
    for xyz1, p1 in items1:
        for xyz2, p2 in items2:
            yield stim.PauliString(xyz1 + xyz2), p1 * p2


def brute_force_rep_code_distill(
        basis: Literal['X', 'Y', 'Z'],
        m1: ErrModel,
        m2: ErrModel | None = None) -> tuple[float, ErrModel]:
    """
    Args:
        basis: Which rep code to use ('X', 'Y', or 'Z').
        m1: Error model of the first qubit.
        m2: Error model of the second qubit, or None (default) to
            use same error model as first qubit.

    Returns:
        A tuple (discard_chance, state_on_success).
    """
    if m2 is None:
        m2 = m1
    if basis == 'X':
        code = rep_xx
    elif basis == 'Y':
        code = rep_yy
    elif basis == 'Z':
        code = rep_zz
    else:
        raise NotImplementedError(f'{basis=}')
    discard = 0
    x = 0
    y = 0
    z = 0
    for err, prob in iter_failures(m1, m2):
        if not all(s.commutes(err) for s in code.stabilizers):
            discard += prob
            continue
        assert len(code.obs_xs) == len(code.obs_zs) == 1
        flip_x = not code.obs_xs[0].commutes(err)
        flip_z = not code.obs_zs[0].commutes(err)
        if flip_x and flip_z:
            y += prob
        elif flip_x:
            z += prob
        elif flip_z:
            x += prob
        else:
            # Not an error.
            pass

    lossage = 1 / (1 - discard)
    return discard, ErrModel(
        x=x * lossage,
        y=y * lossage,
        z=z * lossage,
    )


def distill_x_detect_probability(w1, x1, y1, z1, w2, x2, y2, z2):
    a1 = w1 + x1
    a2 = w2 + x2
    b1 = y1 + z1
    b2 = y2 + z2
    return (a1*b2 + a2*b1) / ((a1 + b1) * (a2 + b2))


def distill_y_detect_probability(w1, x1, y1, z1, w2, x2, y2, z2):
    a1 = w1 + y1
    a2 = w2 + y2
    b1 = x1 + z1
    b2 = x2 + z2
    return (a1*b2 + a2*b1) / ((a1 + b1) * (a2 + b2))


def distill_z_detect_probability(w1, x1, y1, z1, w2, x2, y2, z2):
    a1 = w1 + z1
    a2 = w2 + z2
    b1 = y1 + x1
    b2 = y2 + x2
    return (a1*b2 + a2*b1) / ((a1 + b1) * (a2 + b2))


def distill_x_raw(w1, x1, y1, z1, w2, x2, y2, z2):
    w = w1*w2 + x1*x2
    x = w1*x2 + x1*w2
    y = y1*y2 + z1*z2
    z = y1*z2 + z1*y2
    return w, x, y, z


def distill_y_raw(w1, x1, y1, z1, w2, x2, y2, z2):
    w = w1*w2 + y1*y2
    x = x1*z2 + z1*x2
    y = w1*y2 + y1*w2
    z = x1*x2 + z1*z2
    return w, x, y, z


def distill_z_raw(w1, x1, y1, z1, w2, x2, y2, z2):
    w = w1*w2 + z1*z2
    x = x1*y2 + y1*x2
    y = x1*x2 + y1*y2
    z = w1*z2 + z1*w2
    return w, x, y, z


def distill_x(a: ErrModel, b: ErrModel) -> ErrModel:
    w, x, y, z = distill_x_raw(a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z)
    return ErrModel.from_odds(w, x, y, z)


def distill_y(a: ErrModel, b: ErrModel) -> ErrModel:
    w, x, y, z = distill_y_raw(a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z)
    return ErrModel.from_odds(w, x, y, z)


def distill_z(a: ErrModel, b: ErrModel) -> ErrModel:
    w, x, y, z = distill_z_raw(a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z)
    return ErrModel.from_odds(w, x, y, z)


def compare_x_circuit() -> stim.Circuit:
    return stim.Circuit("""
        SQRT_X 1
        CX 1 0
        H 1
        M 1
    """)


def compare_y_circuit() -> stim.Circuit:
    return stim.Circuit("""
        CY 1 0
        SQRT_X 1
        M 1
    """)


def compare_z_circuit() -> stim.Circuit:
    return stim.Circuit("""
        CY 0 1
        M 1
    """)


def distill_circuit(basis: str) -> stim.Circuit:
    if basis == 'X':
        c = compare_x_circuit()
    elif basis == 'Y':
        c = compare_y_circuit()
    elif basis == 'Z':
        c = compare_z_circuit()
    else:
        raise NotImplementedError(f'{basis=}')
    d = stim.Circuit("""
        R 0 1 2 3
        H 0 1
        CX 0 2 1 3
    """)
    for inst in c:
        d.append(inst)
        d.append(inst.name, [e.qubit_value + 2 for e in inst.targets_copy()])

    if basis in 'XZ':
        d.append(basis, [0])
    return d
