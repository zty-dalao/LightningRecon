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
    # 投影闭环只作为弱物理正则。默认值对应 Stage 2；各阶段根据历史
    # raw loss 的量级单独校准，使两项投影损失合计约占 total 的 5%。
    input_projection: float = 0.0025
    heldout_projection: float = 0.00125

    @classmethod
    def stage1(cls) -> "DualDomainLossWeights":
        """高视角 latent 对齐阶段：强调基础体积和教师蒸馏。"""
        return cls(
            base_image=0.30,
            latent_distillation=0.20,
            residual_regularization=0.02,
            input_projection=0.004,
            heldout_projection=0.002,
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
            input_projection=0.0025,
            heldout_projection=0.00125,
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
            input_projection=0.002,
            heldout_projection=0.001,
        )


class AnatomyPriorLoss(nn.Module):
    """单独预训练 CT-VQ 解剖先验时使用的损失。"""

    def __init__(
        self,
        image_weight: float = 1.0,
        laplacian_weight: float = 0.05,
        structural_weight: float = 0.10,
        vq_weight: float = 0.05,
        boundary_edge_weight: float = 0.05,
    ):
        super().__init__()
        self.image_weight = float(image_weight)
        self.laplacian_weight = float(laplacian_weight)
        self.structural_weight = float(structural_weight)
        self.vq_weight = float(vq_weight)
        self.boundary_edge_weight = float(boundary_edge_weight)
        if self.boundary_edge_weight < 0.0:
            raise ValueError("boundary_edge_weight cannot be negative")

    @staticmethod
    def _edge_magnitude(volume: torch.Tensor) -> torch.Tensor:
        """计算三个方向的平均绝对一阶差分，并保持输入形状。"""
        grad_d = F.pad(
            (volume[:, :, 1:] - volume[:, :, :-1]).abs(),
            (0, 0, 0, 0, 0, 1),
        )
        grad_h = F.pad(
            (volume[:, :, :, 1:] - volume[:, :, :, :-1]).abs(),
            (0, 0, 0, 1, 0, 0),
        )
        grad_w = F.pad(
            (volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1]).abs(),
            (0, 1, 0, 0, 0, 0),
        )
        return (grad_d + grad_h + grad_w) / 3.0

    @staticmethod
    def _balanced_edge_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """平衡稀疏真实边缘，避免辅助头退化为全零预测。"""
        spatial_dims = tuple(range(2, target.ndim))
        mean_edge = target.mean(dim=spatial_dims, keepdim=True).clamp_min(1e-4)
        weights = (1.0 + 4.0 * target / mean_edge).clamp(max=10.0)
        robust_error = torch.sqrt((prediction - target).square() + eps**2)
        return (weights * robust_error).sum() / weights.sum().clamp_min(1.0)

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
        if "boundary_edge" not in outputs:
            raise KeyError("Phase A outputs are missing boundary_edge")
        boundary_edge = self._balanced_edge_loss(
            outputs["boundary_edge"], self._edge_magnitude(target)
        )
        total = (
            self.image_weight * image
            + self.laplacian_weight * laplacian
            + self.structural_weight * structure
            + self.vq_weight * vq
            + self.boundary_edge_weight * boundary_edge
        )
        return {
            "total": total,
            "image": image,
            "laplacian": laplacian,
            "structural": structure,
            "vq": vq,
            "boundary_edge": boundary_edge,
            "weighted_boundary_edge": (
                self.boundary_edge_weight * boundary_edge
            ),
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

        # 系数只能给出预期占比，raw loss 会随训练变化，因此直接记录每个
        # batch 中投影项对 total 的真实贡献。detach 避免诊断量参与反传。
        projection_weighted = (
            weighted["input_projection"]
            + weighted["heldout_projection"]
        )
        projection_fraction = (
            projection_weighted.detach()
            / total.detach().abs().clamp_min(1e-12)
        )

        result = {"total": total}
        result.update({f"raw/{name}": value for name, value in raw.items()})
        result.update(
            {f"weighted/{name}": value for name, value in weighted.items()}
        )
        result["diagnostic/projection_fraction"] = projection_fraction
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
        self.set_gradient_weight(gradient_weight)

    def set_gradient_weight(self, gradient_weight: float) -> None:
        """Update the Phase B edge coefficient with basic validation."""
        gradient_weight = float(gradient_weight)
        if gradient_weight < 0.0:
            raise ValueError("gradient_weight must be non-negative")
        self.gradient_weight = gradient_weight

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
