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
    AnatomyPriorLoss,
    DualDomainLoss,
    DualDomainReconstructionModel,
    HierarchicalAnatomyPrior,
    LearnedForwardProjector,
)
from src.dual_domain.losses import DualDomainLossWeights
from src.thorax_fast_dataset import (
    ThoraxFastDataset,
    resolve_thorax_fast_root,
)
from src.train_phase_c_reconstruction import build_schedule, stage_for_epoch
from src.train_phase_b_projector import (
    gradient_weight_for_epoch,
    learning_rate_for_epoch,
    phase_b_stage,
    projection_visuals,
)
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
            self.assertEqual(resolve_thorax_fast_root(root), root.resolve())
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
        self.assertEqual(
            tuple(outputs["highres_features"].shape),
            (1, 8, 32, 32, 32),
        )
        self.assertEqual(
            tuple(outputs["residual_logits"].shape),
            (1, 1, 32, 32, 32),
        )

        losses = DualDomainLoss()(outputs, teacher_ct)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertIn("diagnostic/projection_fraction", losses)
        self.assertEqual(
            float(losses["diagnostic/projection_fraction"]), 0.0
        )

    def test_projection_cycle_weights_are_weak_stage_regularizers(self):
        expected = {
            1: (0.004, 0.002),
            2: (0.0025, 0.00125),
            3: (0.002, 0.001),
        }
        for stage, (input_weight, heldout_weight) in expected.items():
            weights = getattr(DualDomainLossWeights, f"stage{stage}")()
            self.assertEqual(weights.input_projection, input_weight)
            self.assertEqual(weights.heldout_projection, heldout_weight)

    def test_global_anatomy_and_supervised_boundary_shapes(self):
        prior = HierarchicalAnatomyPrior(
            anatomy_codebook_size=8,
            boundary_codebook_size=4,
            anatomy_dim=8,
            boundary_dim=4,
            base_channels=4,
            prior_feature_channels=4,
            boundary_residual_blocks=3,
            anatomy_transformer_layers=2,
            anatomy_transformer_heads=2,
            anatomy_context_size=2,
            boundary_context_channels=2,
            kmeans_init_batches=1,
        ).eval()
        ct = torch.rand(1, 1, 16, 16, 16)
        with torch.no_grad():
            outputs = prior(ct, update_codebook=False)
        self.assertEqual(
            tuple(outputs["anatomy_latent"].shape), (1, 8, 4, 4, 4)
        )
        self.assertEqual(
            tuple(outputs["boundary_latent"].shape), (1, 4, 8, 8, 8)
        )
        self.assertEqual(
            tuple(outputs["boundary_edge"].shape), (1, 1, 8, 8, 8)
        )
        self.assertEqual(len(prior.encoder.anatomy_transformer.layers), 2)
        losses = AnatomyPriorLoss(boundary_edge_weight=0.05)(outputs, ct)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(float(losses["boundary_edge"]), 0.0)

    def test_high_resolution_checkpoint_path_backpropagates(self):
        model = self._small_model().train()
        projections = torch.randn(1, 2, 1, 16, 16)
        angles = torch.tensor([[-np.pi, 0.0]], dtype=torch.float32)
        outputs = model(projections, angles)
        self.assertNotIn("boundary_edge", outputs)
        outputs["final_volume"].mean().backward()
        gradient = (
            model.sculptor.highres_refinement[0].block[0].weight.grad
        )
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

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
    def test_phase_b_two_stage_lr_and_gradient_schedules(self):
        args = Namespace(
            epochs=250,
            base_epochs=150,
            lr=1e-4,
            edge_lr=2e-5,
            min_lr=1e-6,
            gradient_weight=0.10,
            edge_gradient_weight=0.25,
            edge_gradient_ramp_epochs=25,
        )
        self.assertEqual(phase_b_stage(150, args.base_epochs), 1)
        self.assertEqual(phase_b_stage(151, args.base_epochs), 2)
        self.assertAlmostEqual(learning_rate_for_epoch(1, args), 1e-4)
        self.assertAlmostEqual(learning_rate_for_epoch(150, args), 1e-6)
        self.assertAlmostEqual(learning_rate_for_epoch(151, args), 2e-5)
        self.assertAlmostEqual(learning_rate_for_epoch(250, args), 1e-6)
        self.assertAlmostEqual(gradient_weight_for_epoch(150, args), 0.10)
        self.assertAlmostEqual(gradient_weight_for_epoch(151, args), 0.10)
        self.assertAlmostEqual(gradient_weight_for_epoch(175, args), 0.25)
        self.assertAlmostEqual(gradient_weight_for_epoch(250, args), 0.25)

    def test_phase_b_projection_visuals_are_bounded_images(self):
        prediction = torch.tensor(
            [[[[[-1.0, 0.0], [0.5, 1.0]]]]], dtype=torch.float32
        )
        target = torch.zeros_like(prediction)
        visuals = projection_visuals(prediction, target)
        self.assertEqual(
            set(visuals),
            {
                "target",
                "prediction",
                "absolute_error",
                "target_gradient",
                "prediction_gradient",
                "gradient_error",
            },
        )
        for image in visuals.values():
            self.assertEqual(tuple(image.shape), (1, 2, 2))
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)

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
            composite = save_checkpoint(
                {"epoch": 1},
                run_dir,
                phase="B",
                version="v3",
                kind="best_composite",
            )
            self.assertEqual(
                composite.name, "phase_B_best_composite_v3.pth"
            )
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
