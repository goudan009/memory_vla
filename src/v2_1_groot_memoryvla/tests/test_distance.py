import math

import pytest
import torch

from v2_1_groot_memoryvla.model.memory import Banks, Slot


def configuration():
    q = torch.zeros(82)
    q[3:6] = 1
    q[10] = 1
    return q


def distance(lo, hi, a, b):
    visual = torch.ones(3, 4)
    bank = Banks(2, .2, lo, hi)
    return bank.distance(Slot(0, 0, 1, visual, a), Slot(1, 1, 1, visual, b))


def test_constant_angles_do_not_dilute_valid_translation():
    lo = torch.zeros(82)
    hi = lo.clone()
    hi[0] = 1
    a = configuration()
    b = a.clone()
    b[0] = 1
    assert distance(lo, hi, a, b).item() == pytest.approx(.2)


def test_all_constant_physical_fields_have_zero_distance():
    lo = torch.zeros(82)
    a = configuration()
    b = a.clone()
    b[0] = 1
    result = distance(lo, lo, a, b)
    assert torch.isfinite(result)
    assert result.item() == pytest.approx(0)


@pytest.mark.parametrize('axis', ['yaw', 'roll', 'pitch'])
def test_valid_periodic_angle_uses_shortest_difference(axis):
    lo = torch.zeros(82)
    hi = lo.clone()
    a = configuration()
    b = a.clone()
    first, second = math.pi - .1, -math.pi + .1
    if axis == 'yaw':
        lo[9], hi[9] = -math.pi, math.pi
        a[9], b[9] = first, second
    else:
        cos_index, sin_index = (3, 6) if axis == 'roll' else (4, 7)
        # A varying sine alone is enough to establish angle validity.
        lo[sin_index], hi[sin_index] = -1, 1
        a[cos_index], a[sin_index] = math.cos(first), math.sin(first)
        b[cos_index], b[sin_index] = math.cos(second), math.sin(second)
    assert distance(lo, hi, a, b).item() == pytest.approx(.2 * .2 / math.pi, abs=1e-7)
