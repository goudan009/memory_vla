"""CPU contract tests for the Task40 directed-restore render barrier."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import task40_paired_sim


class Renderer:
    def __init__(self):
        self.block = False
        self.calls = []

    def get_block_on_render(self):
        return self.block

    def set_block_on_render(self, value):
        self.calls.append(value)
        self.block = value


class RestoreRenderTest(unittest.TestCase):
    def test_directed_restore_enables_kit_render_barrier(self):
        sim = Renderer()
        previous = task40_paired_sim.configure_restore_rendering(sim, targeted=True)
        self.assertFalse(previous)
        self.assertTrue(sim.get_block_on_render())
        self.assertEqual(sim.calls, [True])

    def test_fresh_collection_does_not_change_kit_rendering(self):
        sim = Renderer()
        task40_paired_sim.configure_restore_rendering(sim, targeted=False)
        self.assertEqual(sim.calls, [])

    def test_refuses_to_label_unapplied_setting_as_synchronized(self):
        sim = Renderer()
        sim.set_block_on_render = sim.calls.append
        with self.assertRaisesRegex(RuntimeError, "render barrier"):
            task40_paired_sim.configure_restore_rendering(sim, targeted=True)


if __name__ == "__main__":
    unittest.main()
