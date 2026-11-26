"""Source-level regression against the audited V1 action implementation."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[3]
def test_all_v21_python_parses():
    for p in (ROOT/'src/v2_1_groot_memoryvla').rglob('*.py'):
        ast.parse(p.read_text(),filename=str(p))
def test_velocity_insertion_sites():
    text=(ROOT/'src/v2_1_groot_memoryvla/model/velocity.py').read_text()
    assert text.index('u0 = pred_actions')<text.index('pred_actions, delta = refiner')<text.index('action_loss = F.mse_loss')
    assert text.index('u0 = pred_velocity')<text.index('pred_velocity, delta = refiner')<text.index('actions = actions + dt * pred_velocity')
    assert 'refiner(noisy_trajectory, u0, t[:,0,0], cache, robot, 1.)' in text
    assert 'refiner(actions, u0, t_cont, cache, robot, scale)' in text
