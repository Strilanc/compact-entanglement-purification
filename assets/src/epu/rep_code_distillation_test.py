import random

import numpy as np

from epu.err_model import ErrModel
from epu.rep_code_distillation import brute_force_rep_code_distill, distill_x_detect_probability, distill_x, \
    distill_y_detect_probability, distill_y, distill_z_detect_probability, distill_z


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
