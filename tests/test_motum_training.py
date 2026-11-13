"""Focused tests for MOTUM mask, loss, split, and checkpoint behavior."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from monai.networks.nets import SegResNet

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))
from medvision_ai.motum_training import (  # noqa: E402
    SPLIT_COUNTS,
    _atomic_torch_save,
    _balanced_motum_loss,
    _binary_foreground,
    _initialize_motum_head,
    _load_step_one_network,
    _motum_dice,
    _motum_loss,
    _network_from_pretrained,
    _split_cases,
)
from medvision_ai.protocol import MODEL_WEIGHT_SHA256  # noqa: E402


class MotumTrainingTests(unittest.TestCase):
    def test_binary_masks_map_any_nonzero_code_to_foreground(self):
        mask = torch.tensor([0, 1, 2], dtype=torch.int16).numpy().reshape(1, 1, 3)
        foreground, codes = _binary_foreground(mask)
        self.assertEqual(foreground.tolist(), [[[False, True, True]]])
        self.assertEqual(codes, [0, 1, 2])

    def test_two_channel_objective_has_finite_loss_and_gradients(self):
        logits = torch.zeros((1, 2, 16, 16, 16), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, 0, 4:8, 4:8, 4:8] = 1
        targets[:, 1, 8:12, 8:12, 8:12] = 1
        loss = _motum_loss(logits, targets)
        loss.backward()
        scores = _motum_dice(logits.detach(), targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(set(scores), {"nceFlair", "ceT1c"})

    def test_original_loss_gradient_direction_separates_foreground_and_background(self):
        logits = torch.zeros((1, 2, 16, 16, 16), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 4:8, 4:8, 4:8] = 1
        _motum_loss(logits, targets).backward()
        self.assertTrue(torch.all(logits.grad[targets.bool()] < 0))
        self.assertTrue(torch.all(logits.grad[~targets.bool()] > 0))

    def test_original_loss_gradient_direction_separates_foreground_and_background(self):
        logits = torch.zeros((1, 2, 16, 16, 16), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 4:8, 4:8, 4:8] = 1
        _motum_loss(logits, targets).backward()
        self.assertTrue(torch.all(logits.grad[targets.bool()] < 0))
        self.assertTrue(torch.all(logits.grad[~targets.bool()] > 0))

    def test_balanced_loss_pushes_foreground_up_and_background_down(self):
        logits = torch.zeros((1, 2, 16, 16, 16), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 4:8, 4:8, 4:8] = 1
        _balanced_motum_loss(logits, targets).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad[targets.bool()] < 0))
        self.assertTrue(torch.all(logits.grad[~targets.bool()] > 0))

    def test_split_is_deterministic_and_patient_disjoint(self):
        cases = [{"patientId": f"synthetic-{index}"} for index in range(29)]
        first = _split_cases(cases, seed=42)
        second = _split_cases(cases, seed=42)
        self.assertEqual(
            {key: [case["patientId"] for case in value] for key, value in first.items()},
            {key: [case["patientId"] for case in value] for key, value in second.items()},
        )
        self.assertEqual({key: len(value) for key, value in first.items()}, SPLIT_COUNTS)
        groups = [{case["patientId"] for case in values} for values in first.values()]
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)))

    def test_pretrained_transfer_replaces_only_output_convolution(self):
        source = SegResNet(
            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16,
            in_channels=4, out_channels=3, dropout_prob=0.2,
        )
        network, compatible_count = _network_from_pretrained(torch, source.state_dict(), "cpu")
        self.assertEqual(compatible_count, len(source.state_dict()) - 2)
        self.assertEqual(network.conv_final[2].conv.weight.shape[0], 2)
        self.assertTrue(torch.equal(network.convInit.conv.weight, source.convInit.conv.weight))
        self.assertTrue(all(parameter.requires_grad for parameter in network.parameters()))

    def test_prior_bias_initialization_starts_at_patch_prevalence(self):
        network = SegResNet(
            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16,
            in_channels=4, out_channels=2, dropout_prob=0.2,
        )
        prevalence = [0.25, 0.1]
        biases = _initialize_motum_head(network, prevalence)
        head = network.conv_final[2].conv
        self.assertTrue(torch.count_nonzero(head.weight) == 0)
        self.assertTrue(torch.allclose(torch.sigmoid(head.bias), torch.tensor(prevalence), atol=1e-6))
        self.assertTrue(all(abs(a-b) < 1e-6 for a,b in zip(biases, torch.logit(torch.tensor(prevalence)).tolist())))

    def test_step_one_checkpoint_strictly_reloads(self):
        source = SegResNet(
            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16,
            in_channels=4, out_channels=2, dropout_prob=0.2,
        )
        with tempfile.TemporaryDirectory() as temp:
            checkpoint_path = Path(temp) / "synthetic-step-one.pt"
            _atomic_torch_save(torch, {
                "schema": "medvision-motum-experimental-checkpoint-v1",
                "model": source.state_dict(),
                "baseCheckpointSha256": MODEL_WEIGHT_SHA256,
            }, checkpoint_path)
            restored, _ = _load_step_one_network(torch, checkpoint_path, "cpu")
            self.assertTrue(all(
                torch.equal(restored.state_dict()[key], source.state_dict()[key])
                for key in source.state_dict()
            ))


if __name__ == "__main__":
    unittest.main()
