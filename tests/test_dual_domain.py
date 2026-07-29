"""新双域模型、前向投影器和 Thorax Fast 加载器的轻量测试。"""

import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.dual_domain import (
    DualDomainLoss,
    DualDomainReconstructionModel,
    LearnedForwardProjector,
)
from src.thorax_fast_dataset import ThoraxFastDataset
from src.train_phase_c_reconstruction import build_schedule, stage_for_epoch
from src.dual_domain.training_utils import (
    accumulation_window_size,
    build_loader,
    capture_loader_generator_state,
    optimizer_step_due,
    restore_loader_generator_state,
    save_checkpoint,
    select_views,
)
from src.ema_codebook import EMAVectorQuantizer


class ThoraxFastDatasetTest(unittest.TestCase):
    def test_real_pickle_formula_and_periodic_endpoint_are_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projection_dir = root / "processed" / "projections"
            ct_dir = root / "processed" / "images" / "ct"
            projection_dir.mkdir(parents=True)
            ct_dir.mkdir(parents=True)
            (root / "splits.json").write_text(
                json.dumps({"train": ["case"]}), encoding="utf-8"
            )

            # 首尾角是同一物理方向，加载后有效视角应从 5 变为 4。
            projections = np.stack(
                [
                    np.full((4, 8), value, dtype=np.uint8)
                    for value in (0, 64, 128, 192, 255)
                ]
            )
            with (projection_dir / "case.pickle").open("wb") as handle:
                pickle.dump(
                    {
                        "projs": projections,
                        "projs_max": 10.0,
                        "angles": np.linspace(
                            -np.pi, np.pi, 5, dtype=np.float32
                        ),
                    },
                    handle,
                )
            nib.save(
                nib.Nifti1Image(
                    np.full((8, 8, 8), 128, dtype=np.uint8),
                    np.eye(4),
                ),
                ct_dir / "case.nii.gz",
            )

            dataset = ThoraxFastDataset(
                root,
                split="train",
                volume_keys=("ct",),
                projection_views=4,
                projection_size=(4, 8),
                volume_size=(8, 8, 8),
            )
            sample = dataset[0]
            self.assertEqual(int(sample["raw_source_views"]), 5)
            self.assertEqual(int(sample["source_views"]), 4)
            self.assertTrue(bool(sample["duplicate_endpoint_dropped"]))
            self.assertEqual(tuple(sample["projs"].shape), (4, 1, 4, 8))
            self.assertAlmostEqual(float(sample["projs"][0].mean()), -1.0)
            self.assertGreater(float(sample["projs"][-1].mean()), 0.0)
            self.assertEqual(tuple(sample["ct"].shape), (1, 8, 8, 8))


class DualDomainModelTest(unittest.TestCase):
    def _small_model(self):
        return DualDomainReconstructionModel(
            anatomy_codebook_size=8,
            boundary_codebook_size=4,
            anatomy_dim=8,
            boundary_dim=4,
            prior_feature_channels=4,
            refinement_channels=4,
            transformer_layers=1,
            projection_seed_size=2,
            output_size=(32, 32, 32),
        )

    def test_projection_to_base_and_final_volume_shapes(self):
        model = self._small_model().eval()
        projections = torch.randn(1, 2, 1, 16, 16)
        angles = torch.tensor([[-np.pi, 0.0]], dtype=torch.float32)
        teacher_ct = torch.rand(1, 1, 32, 32, 32)

        with torch.no_grad():
            outputs = model(
                projections, angles, teacher_ct=teacher_ct
            )
        self.assertEqual(
            tuple(outputs["base_volume"].shape), (1, 1, 16, 16, 16)
        )
        self.assertEqual(
            tuple(outputs["final_volume"].shape), (1, 1, 32, 32, 32)
        )
        self.assertGreaterEqual(float(outputs["final_volume"].min()), 0.0)
        self.assertLessEqual(float(outputs["final_volume"].max()), 1.0)
        self.assertIn("teacher_anatomy_latent", outputs)

        losses = DualDomainLoss()(outputs, teacher_ct)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_forward_projector_preserves_shape_range_and_gradient(self):
        projector = LearnedForwardProjector(
            projection_size=(16, 16),
            integration_size=8,
            correction_channels=4,
        )
        volume = torch.rand(1, 1, 16, 16, 16, requires_grad=True)
        angles = torch.tensor(
            [[-np.pi, 0.0, np.pi / 2]], dtype=torch.float32
        )
        projections = projector(volume, angles)
        self.assertEqual(tuple(projections.shape), (1, 3, 1, 16, 16))
        self.assertGreaterEqual(float(projections.detach().min()), -1.0)
        self.assertLessEqual(float(projections.detach().max()), 1.0)
        projections.mean().backward()
        self.assertIsNotNone(volume.grad)
        self.assertTrue(torch.isfinite(volume.grad).all())


class TrainingEntrypointTest(unittest.TestCase):
    def test_kmeans_reservoir_accepts_sub_codebook_batch_chunks(self):
        # 32个reservoir样本分8批收集时，每批只取4个，小于8个码字。
        # 单批不应报错；第8批累计样本充足后才启动K-means。
        quantizer = EMAVectorQuantizer(
            n_embed=8,
            embedding_dim=3,
            kmeans_iters=2,
            kmeans_samples_per_code=4,
            kmeans_init_batches=8,
        ).train()
        features = torch.randn(32, 3)
        for batch_index in range(8):
            quantized, loss, perplexity = quantizer(features)
            self.assertEqual(quantized.shape, features.shape)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(perplexity))
            self.assertEqual(
                bool(quantizer.kmeans_initialized.item()),
                batch_index == 7,
            )
        self.assertEqual(int(quantizer.kmeans_reservoir_count.item()), 32)

    def test_phase_c_default_schedule_and_stage_boundaries(self):
        args = Namespace(
            final_view=6,
            view_schedule=None,
            stage1_epochs=2,
            stage2_epochs_per_view=3,
            stage3_epochs=4,
        )
        schedule = build_schedule(args)
        self.assertEqual(
            schedule,
            [
                (1, 2, 60, 1),
                (3, 5, 54, 2),
                (6, 8, 48, 2),
                (9, 11, 36, 2),
                (12, 14, 24, 2),
                (15, 17, 12, 2),
                (18, 20, 6, 2),
                (21, 24, 6, 3),
            ],
        )
        self.assertEqual(stage_for_epoch(schedule, 2), (1, 60))
        self.assertEqual(stage_for_epoch(schedule, 18), (2, 6))
        self.assertEqual(stage_for_epoch(schedule, 24), (3, 6))

    def test_tail_accumulation_and_checkpoint_names(self):
        self.assertEqual(accumulation_window_size(4, 6, 4), 2)
        self.assertFalse(optimizer_step_due(4, 6, 4))
        self.assertTrue(optimizer_step_due(5, 6, 4))
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            best = save_checkpoint(
                {"epoch": 1},
                run_dir,
                phase="C",
                version="v3",
                kind="best",
            )
            final_view = 6
            self.assertEqual(best.name, "phase_C_best_v3.pth")
            self.assertTrue(best.is_file())
            self.assertEqual(final_view, 6)

    def test_fixed_final_views_are_selected_from_base_grid(self):
        projections = torch.arange(60.0).view(1, 60, 1, 1, 1)
        angles = torch.arange(60.0).view(1, 60)
        selected, selected_angles, indices = select_views(
            projections, angles, 6, random_subset=False
        )
        expected = torch.tensor([0, 10, 20, 30, 40, 50])
        self.assertTrue(torch.equal(indices.cpu(), expected))
        self.assertTrue(
            torch.equal(selected.flatten().cpu(), expected.float())
        )
        self.assertTrue(
            torch.equal(selected_angles.flatten().cpu(), expected.float())
        )

    def test_loader_generator_state_can_resume_shuffle_sequence(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(12))
        device = torch.device("cpu")
        loader = build_loader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
            seed=123,
            device=device,
        )
        list(loader)
        state = capture_loader_generator_state(loader)
        expected_next_epoch = [
            batch[0].clone() for batch in loader
        ]

        resumed = build_loader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
            seed=999,
            device=device,
        )
        restore_loader_generator_state(resumed, state)
        actual_next_epoch = [batch[0].clone() for batch in resumed]
        self.assertEqual(len(expected_next_epoch), len(actual_next_epoch))
        for expected, actual in zip(
            expected_next_epoch, actual_next_epoch
        ):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
