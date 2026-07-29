"""双域模型的体素、先验、蒸馏、雕刻和投影闭环损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import (
    charbonnier_loss,
    laplacian_pyramid_loss,
    structural_loss,
)


@dataclass(frozen=True)
class DualDomainLossWeights:
    """主重建阶段的一组可序列化损失权重。"""

    final_image: float = 1.0
    laplacian: float = 0.05
    structural: float = 0.05
    base_image: float = 0.20
    latent_distillation: float = 0.10
    residual_regularization: float = 0.01
    vq: float = 0.0
    input_projection: float = 0.10
    heldout_projection: float = 0.05

    @classmethod
    def stage1(cls) -> "DualDomainLossWeights":
        """高视角 latent 对齐阶段：强调基础体积和教师蒸馏。"""
        return cls(
            base_image=0.30,
            latent_distillation=0.20,
            residual_regularization=0.02,
            input_projection=0.05,
            heldout_projection=0.02,
        )

    @classmethod
    def stage2(cls) -> "DualDomainLossWeights":
        """逐步稀疏阶段：增强投影闭环和结构恢复。"""
        return cls(
            laplacian=0.04,
            structural=0.08,
            base_image=0.20,
            latent_distillation=0.10,
            residual_regularization=0.01,
            input_projection=0.10,
            heldout_projection=0.05,
        )

    @classmethod
    def stage3(cls) -> "DualDomainLossWeights":
        """最终厂家协议微调：主要优化最终体积和实测投影一致性。"""
        return cls(
            laplacian=0.02,
            structural=0.05,
            base_image=0.10,
            latent_distillation=0.05,
            residual_regularization=0.005,
            input_projection=0.10,
            heldout_projection=0.05,
        )


class AnatomyPriorLoss(nn.Module):
    """单独预训练 CT-VQ 解剖先验时使用的损失。"""

    def __init__(
        self,
        image_weight: float = 1.0,
        laplacian_weight: float = 0.05,
        structural_weight: float = 0.10,
        vq_weight: float = 0.05,
    ):
        super().__init__()
        self.image_weight = float(image_weight)
        self.laplacian_weight = float(laplacian_weight)
        self.structural_weight = float(structural_weight)
        self.vq_weight = float(vq_weight)

    def forward(
        self, outputs: dict[str, torch.Tensor], target_ct: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        base = outputs["base_volume"]
        target = F.interpolate(
            target_ct,
            size=base.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        image = charbonnier_loss(base, target)
        laplacian = laplacian_pyramid_loss(base, target)
        structure = structural_loss(base, target)
        vq = outputs["vq_loss"]
        total = (
            self.image_weight * image
            + self.laplacian_weight * laplacian
            + self.structural_weight * structure
            + self.vq_weight * vq
        )
        return {
            "total": total,
            "image": image,
            "laplacian": laplacian,
            "structural": structure,
            "vq": vq,
        }


class DualDomainLoss(nn.Module):
    """组合各项损失，并同时返回 raw/weighted 标量供 TensorBoard 记录。"""

    def __init__(
        self,
        weights: DualDomainLossWeights | None = None,
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        self.weights = weights or DualDomainLossWeights()
        self.charbonnier_eps = float(charbonnier_eps)

    def set_weights(self, weights: DualDomainLossWeights) -> None:
        self.weights = weights

    @staticmethod
    def _zero_like(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_ct: torch.Tensor,
        *,
        input_projections: torch.Tensor | None = None,
        reconstructed_input_projections: torch.Tensor | None = None,
        heldout_projections: torch.Tensor | None = None,
        reconstructed_heldout_projections: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        final_volume = outputs["final_volume"]
        if final_volume.shape != target_ct.shape:
            raise ValueError(
                "final_volume 与 target_ct 必须尺寸一致，"
                f"实际 {final_volume.shape} vs {target_ct.shape}"
            )

        final_image = charbonnier_loss(
            final_volume, target_ct, self.charbonnier_eps
        )
        laplacian = laplacian_pyramid_loss(final_volume, target_ct)
        structure = structural_loss(final_volume, target_ct)

        # 基础体积只监督同分辨率低频目标，不进行先降采样再放大。
        base_volume = outputs["base_volume"]
        base_target = F.interpolate(
            target_ct,
            size=base_volume.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        base_image = charbonnier_loss(
            base_volume, base_target, self.charbonnier_eps
        )

        latent_distillation = self._zero_like(final_volume)
        if "teacher_anatomy_latent" in outputs:
            latent_distillation = 0.5 * (
                F.smooth_l1_loss(
                    outputs["anatomy_latent"],
                    outputs["teacher_anatomy_latent"],
                )
                + F.smooth_l1_loss(
                    outputs["boundary_latent"],
                    outputs["teacher_boundary_latent"],
                )
            )

        # gate 后的有效 residual 才是模型真正施加到基础体积上的修改量。
        residual_regularization = (
            outputs["gate"] * outputs["residual_logits"]
        ).abs().mean()
        vq = outputs.get("vq_loss", self._zero_like(final_volume))

        input_projection = self._zero_like(final_volume)
        if (
            input_projections is not None
            and reconstructed_input_projections is not None
        ):
            input_projection = charbonnier_loss(
                reconstructed_input_projections,
                input_projections,
                self.charbonnier_eps,
            )

        heldout_projection = self._zero_like(final_volume)
        if (
            heldout_projections is not None
            and reconstructed_heldout_projections is not None
        ):
            heldout_projection = charbonnier_loss(
                reconstructed_heldout_projections,
                heldout_projections,
                self.charbonnier_eps,
            )

        raw = {
            "final_image": final_image,
            "laplacian": laplacian,
            "structural": structure,
            "base_image": base_image,
            "latent_distillation": latent_distillation,
            "residual_regularization": residual_regularization,
            "vq": vq,
            "input_projection": input_projection,
            "heldout_projection": heldout_projection,
        }
        weighted = {
            name: value * float(getattr(self.weights, name))
            for name, value in raw.items()
        }
        total = sum(weighted.values())

        result = {"total": total}
        result.update({f"raw/{name}": value for name, value in raw.items()})
        result.update(
            {f"weighted/{name}": value for name, value in weighted.items()}
        )
        return result


class ForwardProjectorLoss(nn.Module):
    """单独训练体素→投影模型时使用的像素与二维边缘损失。"""

    def __init__(
        self,
        image_weight: float = 1.0,
        gradient_weight: float = 0.10,
    ):
        super().__init__()
        self.image_weight = float(image_weight)
        self.gradient_weight = float(gradient_weight)

    @staticmethod
    def _gradient_loss(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
        true_y = target[..., 1:, :] - target[..., :-1, :]
        pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
        true_x = target[..., :, 1:] - target[..., :, :-1]
        return 0.5 * (
            F.l1_loss(pred_y, true_y) + F.l1_loss(pred_x, true_x)
        )

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        image = charbonnier_loss(prediction, target)
        gradient = self._gradient_loss(prediction, target)
        return {
            "total": (
                self.image_weight * image
                + self.gradient_weight * gradient
            ),
            "image": image,
            "gradient": gradient,
        }
