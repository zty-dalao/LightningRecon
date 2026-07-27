import json
import os
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.dataset import ThoraxCTDataset
from src.ema_codebook import EMAVectorQuantizer, EMAVectorQuantizer3D
from src.inference import (
    resampled_affine,
    uniform_view_indices as inference_view_indices,
    validate_checkpoint_metadata,
)
from src.losses import ReconstructionLoss, ssim_3d, structural_loss
from src.models import SparseViewReconstruction, ViewTransformer
from src.train import (
    add_angle_encoding,
    build_run_name,
    build_tensorboard_command,
    build_view_schedule,
    capture_rng_state,
    checkpoint_payload,
    checkpoint_name,
    collate_variable_projection_batch,
    evaluate,
    get_stage_config,
    restore_rng_state,
    subsample_projections,
    uniform_view_indices,
)
from src.view_protocol import resolve_view_curriculum


class FixedViewProtocolTest(unittest.TestCase):
    def test_uniform_subsampling_keeps_projections_and_angles_paired(self):
        projs = torch.arange(48).view(1, 48, 1, 1, 1).float()
        angles = torch.linspace(0, 2 * torch.pi, 49)[:-1].unsqueeze(0)

        sampled, sampled_angles = subsample_projections(
            projs, 6, torch.device('cpu'), angles
        )

        expected = torch.tensor([0, 8, 16, 24, 32, 40])
        self.assertTrue(torch.equal(sampled[0, :, 0, 0, 0].long(), expected))
        self.assertTrue(torch.equal(sampled_angles, angles[:, expected]))

    def test_maximum_view_base_contains_fixed_final_protocol(self):
        for target in (6, 8, 10):
            curriculum = resolve_view_curriculum(target)
            base_views = curriculum[0]
            target_indices = set(
                uniform_view_indices(491,target,torch.device('cpu')).tolist()
            )
            base_indices = set(
                uniform_view_indices(
                    491,base_views,torch.device('cpu')
                ).tolist()
            )
            self.assertTrue(target_indices.issubset(base_indices))

    def test_inference_and_training_share_ten_view_protocol(self):
        train_indices = uniform_view_indices(60, 10, torch.device('cpu'))
        infer_indices = inference_view_indices(60, 10, torch.device('cpu'))
        self.assertTrue(torch.equal(train_indices, infer_indices))

    def test_angle_encoding_uses_physical_angles(self):
        projs = torch.zeros(1, 3, 1, 2, 2)
        angles = torch.tensor([[0.0, torch.pi / 2, torch.pi]])

        encoded = add_angle_encoding(projs, angles)

        self.assertTrue(torch.allclose(encoded[0, :, 1, 0, 0], torch.sin(angles[0])))
        self.assertTrue(torch.allclose(encoded[0, :, 2, 0, 0], torch.cos(angles[0])))

    def test_builtin_schedule_is_selected_by_final_view(self):
        for target, expected in (
            (6, [60,54,48,36,24,12,6,6]),
            (8, [64,56,48,32,24,16,8,8]),
            (10, [60,50,40,30,20,10,10]),
        ):
            args = Namespace(
                stage1_epochs=2,
                stage2_epochs_per_view=1,
                stage3_epochs=1,
                final_view=target,
                view_schedule=None,
            )
            schedule = build_view_schedule(args)
            self.assertEqual([item[2] for item in schedule], expected)
            self.assertEqual(schedule[0][3],1)
            self.assertTrue(all(item[3]==2 for item in schedule[1:-1]))
            self.assertEqual(schedule[-1][3],3)

    def test_command_line_schedule_override_is_validated(self):
        args = Namespace(
            stage1_epochs=2,
            stage2_epochs_per_view=1,
            stage3_epochs=1,
            final_view=6,
            view_schedule='60,48,36,24,12,6',
        )
        schedule=build_view_schedule(args)
        self.assertEqual(
            [item[2] for item in schedule],
            [60,48,36,24,12,6,6],
        )
        self.assertEqual(
            resolve_view_curriculum(6,'60,50,24,12,6'),
            (60,50,24,12,6),
        )
        with self.assertRaises(ValueError):
            resolve_view_curriculum(6,'61,50,24,12,6')

    def test_training_random_subset_is_drawn_from_uniform_base(self):
        total = 491
        encoded = torch.arange(total).view(1,total,1,1,1).float()
        angles = torch.linspace(
            0,2*torch.pi,total + 1
        )[:-1].unsqueeze(0)
        torch.manual_seed(123)
        sampled, sampled_angles = subsample_projections(
            encoded,54,torch.device('cpu'),angles,
            source_total=total,
            view_indices=torch.arange(total).unsqueeze(0),
            base_views=60,
            random_subset=True,
        )
        selected = sampled[0,:,0,0,0].long()
        base = uniform_view_indices(total,60,torch.device('cpu'))
        self.assertEqual(len(torch.unique(selected)),54)
        self.assertTrue(set(selected.tolist()).issubset(set(base.tolist())))
        self.assertTrue(torch.equal(
            sampled_angles[0],
            angles[0].index_select(0,selected),
        ))
        self.assertTrue(torch.all(selected[1:] > selected[:-1]))

    def test_fixed_eval_subset_is_deterministic_within_base(self):
        total = 491
        encoded = torch.arange(total).view(1,total,1,1,1).float()
        first = subsample_projections(
            encoded,6,torch.device('cpu'),
            source_total=total,
            view_indices=torch.arange(total).unsqueeze(0),
            base_views=60,
            random_subset=False,
        )
        second = subsample_projections(
            encoded,6,torch.device('cpu'),
            source_total=total,
            view_indices=torch.arange(total).unsqueeze(0),
            base_views=60,
            random_subset=False,
        )
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(
            first[0,:,0,0,0].long(),
            uniform_view_indices(total,6,torch.device('cpu')),
        ))

    def test_random_view_subset_is_reproducible_after_rng_restore(self):
        total = 491
        encoded = torch.arange(total).view(1,total,1,1,1).float()
        torch.manual_seed(456)
        state = capture_rng_state()
        first = subsample_projections(
            encoded,24,torch.device('cpu'),
            source_total=total,
            view_indices=torch.arange(total).unsqueeze(0),
            base_views=60,
            random_subset=True,
        )
        restore_rng_state(state)
        second = subsample_projections(
            encoded,24,torch.device('cpu'),
            source_total=total,
            view_indices=torch.arange(total).unsqueeze(0),
            base_views=60,
            random_subset=True,
        )
        self.assertTrue(torch.equal(first,second))

    def test_run_and_checkpoint_names_include_final_view_and_version(self):
        self.assertEqual(
            build_run_name('thorax_fast',6,256,'v3'),
            'thorax_fast_finalview=6_256_v3',
        )
        self.assertEqual(
            checkpoint_name('best',6,'v3'),
            'best_model_finalview=6_v3.pth',
        )
        self.assertEqual(
            checkpoint_name('epoch',6,'v3',50),
            'ckpt_0050_finalview=6_v3.pth',
        )
        self.assertEqual(
            checkpoint_name('last',6,'v3'),
            'last_model_finalview=6_v3.pth',
        )
        with self.assertRaises(ValueError):
            build_run_name('thorax_fast',6,256,'../v3')

    def test_transformer_defaults_to_four_encoder_layers(self):
        transformer = ViewTransformer()
        self.assertEqual(len(transformer.transformer.layers), 4)

    def test_tensorboard_command_targets_only_the_versioned_run(self):
        run_name = build_run_name('thorax_fast', 6, 256, 'v3')
        command = build_tensorboard_command(
            os.path.join('logs', run_name, 'tensorboard')
        )
        self.assertIn('thorax_fast_finalview=6_256_v3', command)
        self.assertNotEqual(command, 'tensorboard --logdir "logs"')

    def test_stage2_freezes_ema_and_stage3_freezes_adapters(self):
        self.assertEqual(get_stage_config(1)[2:], (False, False))
        self.assertEqual(get_stage_config(2)[2:], (True, False))
        self.assertEqual(get_stage_config(3)[2:], (True, True))
        self.assertEqual(get_stage_config(1)[0]['w_struct'], 0.05)
        self.assertEqual(get_stage_config(2)[0]['w_struct'], 0.08)
        self.assertEqual(get_stage_config(3)[0]['w_struct'], 0.05)

    def test_frozen_ema_codewords_do_not_update(self):
        quantizer = EMAVectorQuantizer(
            n_embed=8,embedding_dim=2,beta=0.25,decay=0.9,
            kmeans_init_batches=1,
        )
        quantizer.train()
        quantizer(torch.randn(16, 2))
        quantizer.freeze()
        before = quantizer.embedding.weight.detach().clone()
        quantizer(torch.randn(16, 2) + 10)
        self.assertTrue(torch.equal(before, quantizer.embedding.weight))
        diagnostics = quantizer.diagnostics()
        self.assertIn('perplexity', diagnostics)
        self.assertIn('normalized_perplexity', diagnostics)
        self.assertIn('batch_active_fraction', diagnostics)
        self.assertIn('ema_dead_codes', diagnostics)

    def test_kmeans_initializes_once_from_uniform_features(self):
        torch.manual_seed(7)
        quantizer = EMAVectorQuantizer(
            n_embed=8, embedding_dim=2, beta=0.25, decay=0.9,
            kmeans_iters=3, kmeans_samples_per_code=4,
            kmeans_init_batches=1,
            dead_code_warmup_steps=100,
        )
        quantizer.train()
        quantizer(torch.randn(64, 2))
        initialized_weight = quantizer.embedding.weight.detach().clone()
        self.assertTrue(bool(quantizer.kmeans_initialized.item()))
        self.assertEqual(int(quantizer.ema_update_steps.item()), 1)
        quantizer(torch.randn(64, 2) + 20)
        self.assertEqual(int(quantizer.ema_update_steps.item()), 2)
        self.assertFalse(torch.equal(
            initialized_weight, quantizer.embedding.weight
        ))

    def test_dead_codes_are_reinitialized_only_while_ema_updates(self):
        torch.manual_seed(11)
        quantizer = EMAVectorQuantizer(
            n_embed=8, embedding_dim=2, beta=0.25, decay=0.9,
            kmeans_iters=2, kmeans_samples_per_code=2,
            kmeans_init_batches=1,
            dead_code_threshold=0.1,
            dead_code_check_interval=1,
            dead_code_warmup_steps=1,
        )
        quantizer.train()
        quantizer(torch.randn(32, 2))
        quantizer.embedding.cluster_size[:3] = 0
        quantizer.embedding.weight[:3] = 1e6
        quantizer(torch.randn(32, 2))
        diagnostics = quantizer.diagnostics()
        self.assertGreaterEqual(
            int(diagnostics['dead_codes_reinitialized'].item()), 1
        )
        total_before = int(
            quantizer.dead_codes_reinitialized_total.item()
        )
        quantizer.freeze()
        quantizer.embedding.cluster_size[:3] = 0
        quantizer(torch.randn(32, 2))
        self.assertEqual(
            int(quantizer.dead_codes_reinitialized_total.item()),
            total_before,
        )

    def test_model_uses_reduced_default_codebook_sizes(self):
        model = SparseViewReconstruction(n_decoder_ups=1)
        self.assertEqual(model.codebook_hf.codebook.num_tokens, 512)
        self.assertEqual(model.codebook_mf.codebook.num_tokens, 256)

    def test_kmeans_waits_for_multiple_batches_and_tracks_progress(self):
        quantizer = EMAVectorQuantizer(
            n_embed=8,embedding_dim=2,kmeans_iters=2,
            kmeans_samples_per_code=3,kmeans_init_batches=3,
        )
        quantizer.train()
        for expected_progress in (1 / 3, 2 / 3):
            z_q, loss, perplexity = quantizer(torch.randn(32,2))
            self.assertFalse(bool(quantizer.kmeans_initialized.item()))
            self.assertEqual(float(loss),0.0)
            self.assertEqual(float(perplexity),0.0)
            self.assertEqual(z_q.shape,(32,2))
            self.assertAlmostEqual(
                float(quantizer.diagnostics()['kmeans_init_progress']),
                expected_progress,
                places=5,
            )
        quantizer(torch.randn(32,2))
        self.assertTrue(bool(quantizer.kmeans_initialized.item()))
        self.assertEqual(
            float(quantizer.diagnostics()['kmeans_init_progress']),1.0
        )

    def test_empty_kmeans_clusters_are_not_reported_as_active(self):
        quantizer = EMAVectorQuantizer(
            n_embed=8,embedding_dim=2,kmeans_iters=2,
            kmeans_samples_per_code=2,kmeans_init_batches=1,
            dead_code_warmup_steps=100,
        )
        quantizer.train()
        quantizer(torch.zeros(32,2))
        diagnostics = quantizer.diagnostics()
        self.assertEqual(int(diagnostics['batch_active_codes']),1)
        self.assertEqual(int(diagnostics['ema_active_codes']),1)

    def test_codebook_lifecycle_state_survives_state_dict_roundtrip(self):
        source = EMAVectorQuantizer(
            n_embed=8, embedding_dim=2, kmeans_iters=2,
            kmeans_samples_per_code=2,kmeans_init_batches=1,
        )
        source.train()
        source(torch.randn(32, 2))
        source.dead_codes_reinitialized_total.fill_(3)
        restored = EMAVectorQuantizer(
            n_embed=8, embedding_dim=2, kmeans_iters=2,
            kmeans_samples_per_code=2,kmeans_init_batches=1,
        )
        restored.load_state_dict(source.state_dict())
        self.assertTrue(bool(restored.kmeans_initialized.item()))
        self.assertEqual(int(restored.ema_update_steps.item()), 1)
        self.assertEqual(
            int(restored.dead_codes_reinitialized_total.item()), 3
        )

    def test_quantizer_adapters_can_be_frozen_independently(self):
        quantizer = EMAVectorQuantizer3D(n_embed=8,embedding_dim=2)
        quantizer.set_adapter_trainable(False)
        adapter_parameters = list(quantizer.pre_quant.parameters())
        adapter_parameters += list(quantizer.post_quant.parameters())
        self.assertTrue(all(not p.requires_grad for p in adapter_parameters))
        self.assertTrue(quantizer.codebook.embedding._update)


class EvaluationProtocolTest(unittest.TestCase):
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_views = None

        def forward(self, projs):
            self.last_views = projs.shape[1]
            batch = projs.shape[0]
            pred = torch.zeros(batch, 1, 12, 12, 12, device=projs.device)
            return pred, torch.tensor(0.0, device=projs.device), torch.tensor(7.0)

    def test_evaluate_uses_requested_target_views(self):
        model = self.RecordingModel()
        batch = {
            'projs': torch.zeros(1, 48, 1, 2, 2),
            'angles': torch.linspace(0, 2 * torch.pi, 49)[:-1].unsqueeze(0),
            'ct': torch.ones(1, 1, 4, 4, 4),
        }

        metrics = evaluate(
            model,
            [batch],
            torch.device('cpu'),
            ReconstructionLoss(),
            n_views=10,
        )

        self.assertEqual(model.last_views, 10)
        self.assertEqual(metrics['perplexity'], 7.0)

    def test_evaluate_returns_only_whole_volume_metrics(self):
        model = self.RecordingModel()
        batch = {
            'projs': torch.zeros(1,10,1,2,2),
            'angles': torch.linspace(
                0,2*torch.pi,11
            )[:-1].unsqueeze(0),
            'ct': torch.ones(1,1,4,4,4),
        }
        metrics = evaluate(
            model,[batch],torch.device('cpu'),
            ReconstructionLoss(),n_views=6,
        )
        self.assertEqual(
            set(metrics),
            {
                'psnr','ssim','total','img','lap','struct','perplexity'
            },
        )

    def test_full_volume_psnr_is_averaged_per_case(self):
        model = self.RecordingModel()
        ct = torch.empty(2,1,4,4,4)
        ct[0] = 0.5
        ct[1] = 1.0
        batch = {
            'projs': torch.zeros(2,10,1,2,2),
            'angles': torch.linspace(
                0,2*torch.pi,11
            )[:-1].repeat(2,1),
            'ct': ct,
        }
        metrics = evaluate(
            model,[batch],torch.device('cpu'),
            ReconstructionLoss(),n_views=6,
        )
        expected = (
            20*np.log10(1.0/0.5)
            + 20*np.log10(1.0/1.0)
        ) / 2.0
        self.assertAlmostEqual(metrics['psnr'],expected,places=5)


class LossAndGeometryTest(unittest.TestCase):
    def test_structural_loss_compares_edge_locations(self):
        pred = torch.zeros(1, 1, 5, 5, 5)
        ref = torch.zeros_like(pred)
        pred[:, :, 2:, :, :] = 1.0
        ref[:, :, 3:, :, :] = 1.0
        self.assertGreater(float(structural_loss(pred, ref)), 0.0)
        self.assertEqual(float(structural_loss(ref, ref)), 0.0)

    def test_skimage_3d_gaussian_ssim_is_one_for_identical_volumes(self):
        volume = torch.rand(1, 1, 12, 12, 12)
        self.assertAlmostEqual(float(ssim_3d(volume, volume)), 1.0, places=5)
        corrupted = volume.clone()
        corrupted[:, :, 6, 6, 6] = 1.0 - corrupted[:, :, 6, 6, 6]
        self.assertLess(float(ssim_3d(volume, corrupted)), 1.0)

    def test_ssim_matches_per_case_skimage_gaussian_average(self):
        from skimage.metrics import structural_similarity

        torch.manual_seed(3)
        reference = torch.rand(2,1,12,12,12)
        prediction = (reference + 0.05*torch.randn_like(reference)).clamp(0,1)
        expected = np.mean([
            structural_similarity(
                reference[index,0].numpy(),
                prediction[index,0].numpy(),
                data_range=1.0,
                gaussian_weights=True,
                sigma=1.5,
                win_size=11,
                use_sample_covariance=False,
                channel_axis=None,
            )
            for index in range(2)
        ])
        self.assertAlmostEqual(
            float(ssim_3d(prediction,reference)),float(expected),places=6
        )

    def test_resampled_affine_preserves_orientation_and_field_of_view(self):
        reference_affine = np.array([
            [0.0, -2.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        output = resampled_affine(
            reference_affine, (256, 128, 64), (128, 64, 32)
        )
        np.testing.assert_allclose(
            output[:3, :3], reference_affine[:3, :3] * 2.0
        )

    def test_checkpoint_contains_complete_resume_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        args = Namespace(
            run_version='v3', final_view=6,
            source_view_range=(464,491),
            n_decoder_ups=1, transformer_layers=4, proj_size=(128,128),
            hf_codebook_size=512, mf_codebook_size=256,
            kmeans_iters=10, kmeans_samples_per_code=4,
            kmeans_init_batches=8,
            dead_code_threshold=0.1, dead_code_check_interval=100,
            dead_code_warmup_steps=100,
            vol_size=(128,128,128), ct_range=(-1000.0,1000.0),
            stage1_epochs=200,stage2_epochs_per_view=40,
            stage3_epochs=100,batch_size=1,grad_accum=4,amp=True,
        )
        payload = checkpoint_payload(
            epoch=3, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=None, args=args,
            run_name='thorax_finalview=6_256_v3', stage=1,
            current_train_views=60,
            resolved_views=[60,54,48,36,24,12,6],
            best_metric=20.0, best_epoch=3,
        )
        for key in (
            'model_state','optimizer_state','scheduler_state',
            'scaler_state','rng_state','model_config','ct_normalization',
            'training_config',
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload['model_config']['transformer_layers'], 4)
        self.assertEqual(payload['checkpoint_format'], 8)
        self.assertEqual(payload['best_val_psnr'], 20.0)
        self.assertEqual(payload['base_views'], 60)
        self.assertEqual(
            payload['sampling_protocol'],
            'uniform_max_base_then_random_train_subset_fixed_eval',
        )
        self.assertEqual(payload['model_config']['hf_codebook_size'], 512)
        self.assertEqual(payload['model_config']['mf_codebook_size'], 256)
        self.assertEqual(payload['source_view_policy'], 'per_case_actual')
        self.assertEqual(payload['source_view_range'], [464,491])
        self.assertIs(validate_checkpoint_metadata(payload), payload)

    def test_legacy_checkpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata({'model_state': {}})


class ThoraxDatasetTest(unittest.TestCase):
    def test_mixed_source_counts_batch_and_sample_per_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = os.path.join(tmpdir, 'projections')
            ct_dir = os.path.join(tmpdir, 'images', 'ct')
            os.makedirs(proj_dir)
            os.makedirs(ct_dir)
            cases = {'case491': 491, 'case464': 464}
            for case_id, total in cases.items():
                with open(
                    os.path.join(proj_dir, f'{case_id}.pickle'), 'wb'
                ) as handle:
                    pickle.dump({
                        'projs': np.zeros((total,2,2), dtype=np.uint8),
                        'angles': np.linspace(
                            0,2*np.pi,total,endpoint=False
                        ).astype(np.float32),
                        'projs_max': 0.2,
                    }, handle)
                Path(os.path.join(ct_dir, f'{case_id}.nii.gz')).touch()
            with open(os.path.join(tmpdir, 'splits.json'), 'w') as handle:
                json.dump({
                    'train': list(cases),
                    'val': list(cases),
                    'test': list(cases),
                }, handle)
            with patch.object(
                ThoraxCTDataset, '_load_ct',
                return_value=np.zeros((2,2,2), dtype=np.float32),
            ):
                dataset = ThoraxCTDataset(
                    data_root=tmpdir, split='train', n_views=-1,
                    proj_size=(2,2), vol_size=(2,2,2),
                )
                batch = next(iter(torch.utils.data.DataLoader(
                    dataset, batch_size=2, shuffle=False,
                    collate_fn=collate_variable_projection_batch,
                )))

            self.assertEqual(
                [len(item) for item in batch['projs']], [491,464]
            )
            encoded_indices = [
                item.float().view(-1,1,1,1)
                for item in batch['view_indices']
            ]
            sampled, sampled_angles = subsample_projections(
                encoded_indices, 6, torch.device('cpu'), batch['angles'],
                source_total=batch['source_views'],
                view_indices=batch['view_indices'],
            )
            for sample_index, total in enumerate(batch['source_views']):
                expected = uniform_view_indices(
                    int(total),6,torch.device('cpu')
                )
                self.assertTrue(torch.equal(
                    sampled[sample_index,:,0,0,0].long(), expected
                ))
                expected_angles = (
                    expected.float() * (2 * torch.pi / int(total))
                )
                self.assertTrue(torch.allclose(
                    sampled_angles[sample_index],
                    expected_angles,
                    atol=1e-6,
                ))

    def test_dataset_uses_fixed_uniform_indices_and_returns_angles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = os.path.join(tmpdir, 'projections')
            ct_dir = os.path.join(tmpdir, 'images', 'ct')
            os.makedirs(proj_dir)
            os.makedirs(ct_dir)

            total = 49
            projections = np.arange(total, dtype=np.uint8)[:, None, None]
            projections = np.broadcast_to(projections, (total, 8, 16)).copy()
            angles = np.linspace(0, 2 * np.pi, total, endpoint=False).astype(np.float32)
            with open(os.path.join(proj_dir, 'case0.pickle'), 'wb') as handle:
                pickle.dump(
                    {
                        'projs': projections,
                        'angles': angles,
                        'projs_max': 0.2,
                    },
                    handle,
                )

            Path(os.path.join(ct_dir, 'case0.nii.gz')).touch()
            with open(os.path.join(tmpdir, 'splits.json'), 'w') as handle:
                json.dump(
                    {'train': ['case0'], 'val': ['case0'], 'test': ['case0']},
                    handle,
                )

            with patch.object(
                ThoraxCTDataset,
                '_load_ct',
                return_value=np.zeros((4, 4, 4), dtype=np.float32),
            ):
                dataset = ThoraxCTDataset(
                    data_root=tmpdir,
                    split='train',
                    n_views=6,
                    proj_size=(16, 16),
                    vol_size=(4, 4, 4),
                )
                sample = dataset[0]

            expected = np.floor(np.arange(6) * total / 6).astype(np.int64)
            self.assertEqual(sample['projs'].shape, (6, 1, 16, 16))
            self.assertEqual(sample['ct'].shape, (1, 4, 4, 4))
            self.assertEqual(
                set(sample),
                {
                    'case_id','projs','angles','view_indices',
                    'source_views','ct',
                },
            )
            self.assertTrue(
                torch.equal(sample['view_indices'], torch.from_numpy(expected))
            )
            self.assertTrue(
                torch.allclose(sample['angles'], torch.from_numpy(angles[expected]))
            )

    def test_full_source_supports_random_subsets_of_uniform_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = os.path.join(tmpdir, 'projections')
            ct_dir = os.path.join(tmpdir, 'images', 'ct')
            os.makedirs(proj_dir)
            os.makedirs(ct_dir)
            total = 491
            projections = np.zeros((total, 2, 2), dtype=np.uint8)
            angles = np.linspace(
                0, 2 * np.pi, total, endpoint=False
            ).astype(np.float32)
            with open(os.path.join(proj_dir, 'case0.pickle'), 'wb') as handle:
                pickle.dump({
                    'projs': projections,
                    'angles': angles,
                    'projs_max': 0.2,
                }, handle)
            Path(os.path.join(ct_dir, 'case0.nii.gz')).touch()
            with open(os.path.join(tmpdir, 'splits.json'), 'w') as handle:
                json.dump({
                    'train': ['case0'], 'val': ['case0'], 'test': ['case0']
                }, handle)
            with patch.object(
                ThoraxCTDataset, '_load_ct',
                return_value=np.zeros((2,2,2), dtype=np.float32),
            ):
                dataset = ThoraxCTDataset(
                    data_root=tmpdir,split='train',n_views=-1,
                    expected_source_views=491,
                    proj_size=(2,2),vol_size=(2,2,2),
                )
                sample = dataset[0]
            self.assertEqual(len(sample['view_indices']), 491)
            loaded = set(sample['view_indices'].tolist())
            base = uniform_view_indices(
                491,60,torch.device('cpu')
            )
            self.assertTrue(set(base.tolist()).issubset(loaded))
            for count in (60,54,48,36,24,12,6):
                encoded_indices = sample['view_indices'].float().view(
                    1,-1,1,1,1
                )
                selected,_ = subsample_projections(
                    encoded_indices,count,torch.device('cpu'),
                    sample['angles'].unsqueeze(0),
                    source_total=491,
                    view_indices=sample['view_indices'].unsqueeze(0),
                    base_views=60,
                    random_subset=True,
                )
                selected_indices = selected[0,:,0,0,0].long()
                self.assertEqual(len(torch.unique(selected_indices)),count)
                self.assertTrue(
                    set(selected_indices.tolist()).issubset(set(base.tolist()))
                )


if __name__ == '__main__':
    unittest.main()
