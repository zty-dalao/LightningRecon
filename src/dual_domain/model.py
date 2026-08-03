"""投影→基础体积→残差雕刻的主重建模型。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .anatomy_prior import HierarchicalAnatomyPrior
from .projection_encoder import SparseProjectionEncoder
from .refiner import ResidualVolumeSculptor


class DualDomainReconstructionModel(nn.Module):
    """组合 CT 解剖先验、投影 latent 预测和病例特异雕刻器。

    常规推理只运行本模型。体素到投影模型在训练循环外单独创建、预训练并
    冻结，用于对 ``final_volume`` 施加投影域循环一致性。
    """

    def __init__(
        self,
        anatomy_prior: HierarchicalAnatomyPrior | None = None,
        *,
        anatomy_codebook_size: int = 512,
        boundary_codebook_size: int = 256,
        anatomy_dim: int = 64,
        boundary_dim: int = 32,
        prior_feature_channels: int = 32,
        refinement_channels: int = 16,
        highres_channels: int = 8,
        checkpoint_highres: bool = True,
        transformer_layers: int = 4,
        projection_seed_size: int = 16,
        output_size: tuple[int, int, int] = (256, 256, 256),
    ):
        super().__init__()
        self.anatomy_prior = anatomy_prior or HierarchicalAnatomyPrior(
            anatomy_codebook_size=anatomy_codebook_size,
            boundary_codebook_size=boundary_codebook_size,
            anatomy_dim=anatomy_dim,
            boundary_dim=boundary_dim,
            prior_feature_channels=prior_feature_channels,
        )
        self.projection_encoder = SparseProjectionEncoder(
            feature_dim=64,
            volume_channels=32,
            anatomy_dim=anatomy_dim,
            boundary_dim=boundary_dim,
            refinement_channels=refinement_channels,
            transformer_layers=transformer_layers,
            seed_size=projection_seed_size,
        )
        self.sculptor = ResidualVolumeSculptor(
            prior_feature_channels=prior_feature_channels,
            projection_feature_channels=refinement_channels,
            hidden_channels=24,
            highres_channels=highres_channels,
            checkpoint_highres=checkpoint_highres,
            output_size=output_size,
        )

    def forward(
        self,
        projections: torch.Tensor,
        angles: torch.Tensor,
        *,
        teacher_ct: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """重建最终体积，并可选返回真实 CT latent 作为蒸馏教师。"""
        predicted = self.projection_encoder(projections, angles)
        quantized = self.anatomy_prior.quantize(
            predicted["anatomy_latent"],
            predicted["boundary_latent"],
            update_codebook=False,
        )
        prior = self.anatomy_prior.decode(
            quantized["anatomy_quantized"],
            quantized["boundary_quantized"],
            compute_boundary_edge=False,
        )
        refined = self.sculptor(
            prior["base_volume"],
            prior["prior_features"],
            predicted["refinement_features"],
        )

        outputs = {**predicted, **quantized, **prior, **refined}
        if teacher_ct is not None:
            # 教师 latent 不需要梯度，但不能对主重建前向整体使用 no_grad。
            with torch.no_grad():
                teacher = self.anatomy_prior.encode_ct(teacher_ct)
            outputs["teacher_anatomy_latent"] = teacher["anatomy_latent"]
            outputs["teacher_boundary_latent"] = teacher["boundary_latent"]
        return outputs

    def freeze_pretrained_prior(
        self, *, freeze_decoder: bool = False
    ) -> None:
        """主训练前冻结 CT Encoder/codebook，可选择是否冻结基础解码器。"""
        self.anatomy_prior.freeze_teacher_encoder()
        self.anatomy_prior.freeze_codebooks()
        for parameter in (
            self.anatomy_prior.decoder.boundary_edge_head.parameters()
        ):
            parameter.requires_grad = False
        if freeze_decoder:
            for parameter in self.anatomy_prior.decoder.parameters():
                parameter.requires_grad = False

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """返回建议的优化器分组，便于设置不同学习率。"""
        return {
            "projection_encoder": [
                parameter
                for parameter in self.projection_encoder.parameters()
                if parameter.requires_grad
            ],
            "prior_decoder": [
                parameter
                for parameter in self.anatomy_prior.decoder.parameters()
                if parameter.requires_grad
            ],
            "sculptor": [
                parameter
                for parameter in self.sculptor.parameters()
                if parameter.requires_grad
            ],
        }
