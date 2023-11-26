import random

import numpy as np
import stim

from epu.err_model import ErrModel
from epu.rep_code_distillation import brute_force_rep_code_distill, distill_x_detect_probability, distill_x, \
    distill_y_detect_probability, distill_y, distill_z_detect_probability, distill_z, compare_x_circuit, \
    compare_y_circuit, compare_z_circuit, distill_circuit
from epu.stabilizer_code import rep_xx, rep_yy, rep_zz


def test_brute_force_distill_versus_analytic_distill():
    w1 = random.random()
    x1 = random.random()
    y1 = random.random()
    z1 = random.random()
    w2 = random.random()
    x2 = random.random()
    y2 = random.random()
    z2 = random.random()
    m1 = ErrModel.from_odds(w1, x1, y1, z1)
    m2 = ErrModel.from_odds(w2, x2, y2, z2)

    pa = distill_x_detect_probability(w1, x1, y1, z1, w2, x2, y2, z2)
    ma = distill_x(m1, m2)
    pb, mb = brute_force_rep_code_distill('X', m1, m2)
    assert np.allclose([pa, ma.x, ma.y, ma.z], [pb, mb.x, mb.y, mb.z], atol=1e-6, rtol=1e-8)

    pa = distill_y_detect_probability(w1, x1, y1, z1, w2, x2, y2, z2)
    ma = distill_y(m1, m2)
    pb, mb = brute_force_rep_code_distill('Y', m1, m2)
    assert np.allclose([pa, ma.x, ma.y, ma.z], [pb, mb.x, mb.y, mb.z], atol=1e-6, rtol=1e-8)

    pa = distill_z_detect_probability(w1, x1, y1, z1, w2, x2, y2, z2)
    ma = distill_z(m1, m2)
    pb, mb = brute_force_rep_code_distill('Z', m1, m2)
    assert np.allclose([pa, ma.x, ma.y, ma.z], [pb, mb.x, mb.y, mb.z], atol=1e-6, rtol=1e-8)


def test_compare_circuits():
    cx = compare_x_circuit()
    assert cx.has_flow(start=rep_xx.stabilizers[0], measurements=[0])
    assert cx.has_flow(start=rep_xx.obs_xs[0], end=stim.PauliString("X_"))
    assert cx.has_flow(start=rep_xx.obs_zs[0], end=stim.PauliString("Z_"))

    cy = compare_y_circuit()
    assert cy.has_flow(start=rep_yy.stabilizers[0], measurements=[0])
    assert cy.has_flow(start=rep_yy.obs_xs[0], end=stim.PauliString("X_"))
    assert cy.has_flow(start=rep_yy.obs_zs[0], end=stim.PauliString("Z_"))

    cz = compare_z_circuit()
    assert cz.has_flow(start=rep_zz.stabilizers[0], measurements=[0])
    assert cz.has_flow(start=rep_zz.obs_xs[0], end=stim.PauliString("X_"))
    assert cz.has_flow(start=rep_zz.obs_zs[0], end=stim.PauliString("Z_"))


def test_distill_circuits():
    dx = distill_circuit('X')
    assert dx.has_flow(end=stim.PauliString('X_X_'))
    assert dx.has_flow(end=stim.PauliString('Z_Z_'))

    dy = distill_circuit('Y')
    assert dy.has_flow(end=stim.PauliString('X_X_'))
    assert dy.has_flow(end=stim.PauliString('Z_Z_'))

    dz = distill_circuit('Z')
    assert dz.has_flow(end=stim.PauliString('X_X_'))
    assert dz.has_flow(end=stim.PauliString('Z_Z_'))
