"""新两阶段模型和嵌套视角协议的轻量CPU测试。"""

import unittest

import torch

from src.fast_sculpt import (
    BaseSCTLoss, BaseSCTNet, ProjectionGuidedSculptor, SculptingLoss,
)
from src.view_protocol import nested_view_indices


class NestedViewsTest(unittest.TestCase):
    def test_final_uniform_anchors_exist_in_every_stage(self):
        final = set(nested_view_indices(491, 60, 6, 6))
        for views in (60, 54, 48, 36, 24, 12, 6):
            selected = set(nested_view_indices(491, 60, views, 6))
            self.assertEqual(len(selected), views)
            self.assertTrue(final.issubset(selected))


class FastSculptModelTest(unittest.TestCase):
    def test_two_stage_shapes_and_gradients(self):
        cbct = torch.rand(1, 1, 32, 32, 32)
        ct = torch.rand_like(cbct)
        base_model = BaseSCTNet(base_channels=2)
        base_outputs = base_model(cbct)
        self.assertEqual(base_outputs["base_sct"].shape, cbct.shape)
        BaseSCTLoss()(base_outputs, ct)["total"].backward()

        sculptor = ProjectionGuidedSculptor(
            base_channels=2, projection_channels=8, evidence_size=4
        )
        projections = torch.rand(1, 6, 1, 32, 32) * 2 - 1
        angles = torch.linspace(-3.14159, 3.14159, 7)[:-1].unsqueeze(0)
        outputs = sculptor(
            base_outputs["base_sct"].detach(), projections, angles
        )
        self.assertEqual(outputs["final_sct"].shape, cbct.shape)
        self.assertEqual(outputs["evidence_gate"].shape, cbct.shape)
        self.assertEqual(outputs["evidence_gate_logits"].shape, cbct.shape)
        loss = SculptingLoss()(
            outputs, base_outputs["base_sct"].detach(), ct
        )["total"]
        loss.backward()
        self.assertIsNotNone(sculptor.residual.weight.grad)


if __name__ == "__main__":
    unittest.main()
