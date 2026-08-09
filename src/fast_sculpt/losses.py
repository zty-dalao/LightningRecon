"""两阶段训练损失：体素强度为主，结构与边缘为辅。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import charbonnier_loss, laplacian_pyramid_loss, structural_loss


class BaseSCTLoss(nn.Module):
    def __init__(self, image=1.0, laplacian=0.10, structural=0.10,
                 residual=0.002):
        super().__init__()
        self.weights = dict(image=image, laplacian=laplacian,
                            structural=structural, residual=residual)

    def forward(self, outputs, target):
        raw = {
            "image": charbonnier_loss(outputs["base_sct"], target),
            "laplacian": laplacian_pyramid_loss(outputs["base_sct"], target),
            "structural": structural_loss(outputs["base_sct"], target),
            "residual": (
                outputs["base_gate"] * outputs["base_residual"]
            ).abs().mean(),
        }
        weighted = {k: raw[k] * self.weights[k] for k in raw}
        return {"total": sum(weighted.values()), "raw": raw,
                "weighted": weighted}


class SculptingLoss(nn.Module):
    """鼓励有必要处更新，并阻止低变化区域被投影噪声破坏。"""

    def __init__(self, image=1.0, laplacian=0.12, structural=0.15,
                 preserve=0.05, gate=0.02, residual=0.002):
        super().__init__()
        self.weights = dict(image=image, laplacian=laplacian,
                            structural=structural, preserve=preserve,
                            gate=gate, residual=residual)

    def forward(self, outputs, base_sct, target):
        final = outputs["final_sct"]
        # CT与基底差异只在训练时构造软监督，推理不需要CT。
        need_update = (target - base_sct).abs()
        need_update = F.avg_pool3d(need_update, 5, stride=1, padding=2)
        gate_target = (need_update / 0.08).clamp(0.0, 1.0)
        modification = (final - base_sct).abs()
        raw = {
            "image": charbonnier_loss(final, target),
            "laplacian": laplacian_pyramid_loss(final, target),
            "structural": structural_loss(final, target),
            "preserve": ((1.0 - gate_target) * modification).mean(),
            "gate": F.binary_cross_entropy(
                outputs["evidence_gate"].clamp(1e-5, 1 - 1e-5), gate_target
            ),
            "residual": (
                outputs["evidence_gate"] * outputs["sculpt_residual"]
            ).abs().mean(),
        }
        weighted = {k: raw[k] * self.weights[k] for k in raw}
        return {"total": sum(weighted.values()), "raw": raw,
                "weighted": weighted}
