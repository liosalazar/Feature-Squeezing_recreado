"""Faithful PyTorch reimplementation of the paper's MNIST FGSM/1-bit experiment.

This is a scoped reproduction of the robustness row in Table 3, not a full
reproduction of every dataset, attack, and detector reported in the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 1234


class PaperMNISTCNN(nn.Module):
    """CleverHans v1.0 MNIST model_mnist architecture used by Xu et al."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout_in = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 10)
        for layer in (self.conv1, self.conv2, self.conv3, self.fc):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout_in(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x.flatten(1))
        return self.fc(x)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist(device: torch.device):
    train = MNIST(str(DATA_DIR), train=True, download=True)
    test = MNIST(str(DATA_DIR), train=False, download=True)
    x_train = train.data.unsqueeze(1).float().div_(255.0).to(device)
    y_train = train.targets.to(device)
    x_test = test.data.unsqueeze(1).float().div_(255.0).to(device)
    y_test = test.targets.to(device)
    return x_train, y_train, x_test, y_test


def make_model(device: torch.device) -> PaperMNISTCNN:
    model_path = RESULTS_DIR / "mnist_cnn_paper_architecture.pt"
    model = PaperMNISTCNN().to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        return model.eval()

    x_train, y_train, x_test, y_test = load_mnist(device)
    # Paper settings: 100 epochs, batch 128, Adadelta(lr=.1, rho=.95, eps=1e-8).
    # The paper applies 0.1 label smoothing by clipping one-hot labels.
    y_soft = torch.full((len(y_train), 10), 0.1 / 9.0, device=device)
    y_soft.scatter_(1, y_train[:, None], 0.9)
    loader = DataLoader(TensorDataset(x_train, y_soft), batch_size=128, shuffle=False)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, rho=0.95, eps=1e-8)
    model.train()
    history = []
    for epoch in range(100):
        loss_total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = -(yb * F.log_softmax(model(xb), dim=1)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            clean_acc = (model(x_test).argmax(1) == y_test).float().mean().item()
        history.append({"epoch": epoch + 1, "train_loss": loss_total / len(x_train), "test_accuracy": clean_acc})
        print(f"epoch {epoch + 1:03d}/100 loss={history[-1]['train_loss']:.5f} test_acc={clean_acc:.4%}", flush=True)
        model.train()
    model.eval()
    torch.save(model.state_dict(), model_path)
    with (RESULTS_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_accuracy"])
        writer.writeheader()
        writer.writerows(history)
    return model.eval()


@torch.no_grad()
def predict_probs(model: nn.Module, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    parts = [F.softmax(model(x[i:i + batch_size]), dim=1) for i in range(0, len(x), batch_size)]
    return torch.cat(parts, dim=0)


def fgsm_paper(model: nn.Module, x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Generate FGSM examples, then clip to [0,1] and quantize to 8-bit pixels."""
    x_adv = x.detach().clone().requires_grad_(True)
    model.eval()
    logits = model(x_adv)
    pseudo_labels = logits.detach().argmax(dim=1)
    loss = F.cross_entropy(logits, pseudo_labels, reduction="sum")
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = (x + epsilon * grad.sign()).clamp(0.0, 1.0)
    return (x_adv * 255.0).round().div(255.0).detach()


def binary_squeeze(x: torch.Tensor) -> torch.Tensor:
    """Paper's 1-bit filter: round each grayscale pixel to 0 or 1 at 0.5."""
    return (x > 0.5).to(x.dtype)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, clip: bool = False) -> float:
    if clip:
        x = x.clamp(0.0, 1.0)
    pred = predict_probs(model, x).argmax(1)
    return (pred == y).float().mean().item()


def scores_l1(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Paper Eq. (6): L1 distance between class-probability vectors."""
    p_raw = predict_probs(model, x.clamp(0.0, 1.0))
    p_sq = predict_probs(model, binary_squeeze(x))
    return (p_raw - p_sq).abs().sum(dim=1)


def threshold_at_fpr(clean_scores: torch.Tensor, max_fpr: float = 0.05) -> float:
    # Threshold is selected using legitimate calibration data only, as in Sec. 5-B.
    values = torch.sort(clean_scores.flatten()).values
    index = min(int(np.floor((1.0 - max_fpr) * len(values))), len(values) - 1)
    return float(values[index].item())


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    # Mann-Whitney/rank form; no external metrics package is needed.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def save_visual(x: torch.Tensor, x_clean_sq: torch.Tensor, x_adv: torch.Tensor,
                x_adv_sq: torch.Tensor, label: int, pred_clean: int,
                pred_clean_sq: int, pred_adv: int, pred_adv_sq: int,
                clean_score: float, adv_score: float, threshold: float) -> None:
    scale = 7
    tile_size = 28 * scale
    panel = Image.new("RGB", (1120, 430), (246, 248, 251))
    draw = ImageDraw.Draw(panel)
    font_path = r"C:\Windows\Fonts\arial.ttf"
    font_header = ImageFont.truetype(font_path, 23)
    font_title = ImageFont.truetype(font_path, 18)
    font_body = ImageFont.truetype(font_path, 17)
    font_footer = ImageFont.truetype(font_path, 18)
    cards = [
        (24, "Imagen legítima", x, x_clean_sq, pred_clean, pred_clean_sq, clean_score, "ACEPTADA"),
        (568, "Imagen FGSM (ε = 0.3)", x_adv, x_adv_sq, pred_adv, pred_adv_sq, adv_score, "RECHAZADA"),
    ]
    for ox, title, before, after, pred_before, pred_after, score, status in cards:
        draw.rounded_rectangle((ox, 18, ox + 528, 374), radius=14, fill="white", outline=(220, 226, 234), width=2)
        draw.text((ox + 18, 28), title, fill=(24, 39, 61), font=font_header)
        for j, (img, caption, pred) in enumerate([(before, "Original", pred_before), (after, "Tras squeezing 1 bit", pred_after)]):
            px = (img.squeeze().detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            tile = Image.fromarray(px, mode="L").resize((tile_size, tile_size), Image.Resampling.NEAREST).convert("RGB")
            tx = ox + 18 + j * 250
            ty = 68
            panel.paste(tile, (tx, ty))
            draw.text((tx, ty + tile_size + 7), caption, fill=(24, 39, 61), font=font_title)
            draw.text((tx, ty + tile_size + 31), f"Predicción: {pred} | real: {label}", fill=(45, 57, 74), font=font_body)
        relation = "≤" if score <= threshold else ">"
        draw.text((ox + 18, 352), f"Puntaje L1={score:.4f} {relation} umbral {threshold:.4f} → {status}", fill=(45, 57, 74), font=font_body)
    clean_state = "RECHAZADA" if clean_score > threshold else "ACEPTADA"
    adv_state = "RECHAZADA" if adv_score > threshold else "SOSPECHOSA / RECHAZADA"
    draw.text((25, 393), f"La regla compara la salida del mismo clasificador antes y después del filtro. Umbral L1={threshold:.4f}", fill=(18, 39, 72), font=font_footer)
    panel.save(RESULTS_DIR / "mnist_fgsm_feature_squeezing.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-count", type=int, default=1000,
                        help="Number of correctly classified MNIST test images for the detector experiment.")
    args = parser.parse_args()
    set_seed()
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    x_train, y_train, x_test, y_test = load_mnist(device)
    model = make_model(device)

    # The paper's robustness table selects the first 100 correctly classified
    # test images. Its detector experiment uses 1,000 clean and 1,000 adversarial
    # samples, split in half for threshold fitting and validation.
    clean_test_probs = predict_probs(model, x_test)
    correct_indices = torch.where(clean_test_probs.argmax(1) == y_test)[0]
    table_n = min(100, len(correct_indices))
    table_indices = correct_indices[:table_n]
    x_table, y_table = x_test[table_indices], y_test[table_indices]
    adv_table = fgsm_paper(model, x_table, 0.3)
    adv_table_binary = binary_squeeze(adv_table)
    det_n = min(args.sample_count, len(correct_indices))
    det_indices = correct_indices[:det_n]
    x, y = x_test[det_indices], y_test[det_indices]
    x_adv = fgsm_paper(model, x, 0.3)
    x_binary = binary_squeeze(x_adv)
    clean_binary = binary_squeeze(x)
    with torch.no_grad():
        clean_probs = predict_probs(model, x)
        clean_binary_probs = predict_probs(model, clean_binary)
        adv_probs = predict_probs(model, x_adv)
        binary_probs = predict_probs(model, x_binary)
        scores_clean_all = (clean_probs - clean_binary_probs).abs().sum(1)
        scores_adv_all = (adv_probs - binary_probs).abs().sum(1)

    # Separate 50/50 fit and validation portions; threshold uses clean calibration only.
    split = max(1, det_n // 2)
    calibration_scores = scores_clean_all[:split]
    threshold = threshold_at_fpr(calibration_scores, 0.05)
    clean_eval_scores = scores_clean_all[split:]
    adv_eval_scores = scores_adv_all[split:]
    raw_pred = adv_probs.argmax(1)
    successful = raw_pred[split:] != y[split:]
    fp_rate = float((clean_eval_scores > threshold).float().mean().item()) if len(clean_eval_scores) else float("nan")
    det_rate_all_adv = float((adv_eval_scores > threshold).float().mean().item()) if len(adv_eval_scores) else float("nan")
    det_rate_successful = float((adv_eval_scores[successful] > threshold).float().mean().item()) if successful.any() else float("nan")
    auc_labels = np.concatenate([np.zeros(len(clean_eval_scores), dtype=int), np.ones(len(adv_eval_scores), dtype=int)])
    auc_scores = np.concatenate([clean_eval_scores.cpu().numpy(), adv_eval_scores.cpu().numpy()])

    metrics = {
        "article": "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks (NDSS 2018)",
        "scope": "Scoped reproduction of the MNIST FGSM/1-bit experiment and L1 detector. Not a full reproduction of all three datasets and eleven attack variants.",
        "model_clean_accuracy": accuracy(model, x_test, y_test),
        "clean_accuracy_after_1bit": accuracy(model, binary_squeeze(x_test), y_test),
        "fgsm_epsilon": 0.3,
        "table3_first_100_correct_seed_count": table_n,
        "table3_seed_clean_accuracy": accuracy(model, x_table, y_table),
        "table3_seed_fgsm_accuracy_without_squeezing": accuracy(model, adv_table, y_table),
        "table3_seed_fgsm_accuracy_after_1bit_squeezing": accuracy(model, adv_table_binary, y_table),
        "detection_clean_adversarial_pairs": det_n,
        "detection_calibration_count_clean": split,
        "detection_evaluation_count_clean": len(clean_eval_scores),
        "detection_evaluation_count_adversarial": len(adv_eval_scores),
        "detection_threshold_clean_5pct_fpr": threshold,
        "detection_false_positive_rate_clean": fp_rate,
        "detection_rate_all_adversarial": det_rate_all_adv,
        "detection_rate_successful_adversarial": det_rate_successful,
        "detection_roc_auc": roc_auc(auc_scores, auc_labels),
        "device": str(device),
        "seed": SEED,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    # Save one successful example corrected by squeezing and flagged by the detector.
    squeezed_pred = binary_probs.argmax(1)
    candidates = torch.where((clean_probs.argmax(1) == y) & (raw_pred != y) & (squeezed_pred == y) & (scores_adv_all > threshold))[0]
    if not len(candidates):
        candidates = torch.where((clean_probs.argmax(1) == y) & (raw_pred != y))[0]
    idx = int(candidates[0].item()) if len(candidates) else 0
    with torch.no_grad():
        clean_sq_prob = predict_probs(model, binary_squeeze(x[idx:idx + 1]))[0]
        clean_score = float((clean_probs[idx] - clean_sq_prob).abs().sum().item())
        score = float(scores_adv_all[idx].item())
        save_visual(x[idx], binary_squeeze(x[idx]), x_adv[idx], x_binary[idx], int(y[idx].item()),
                    int(clean_probs[idx].argmax().item()), int(clean_sq_prob.argmax().item()),
                    int(raw_pred[idx].item()), int(binary_probs[idx].argmax().item()),
                    clean_score, score, threshold)
        metrics["demo_example"] = {
            "test_index": int(det_indices[idx].item()),
            "true_digit": int(y[idx].item()),
            "clean_prediction": int(clean_probs[idx].argmax().item()),
            "clean_squeezed_prediction": int(clean_sq_prob.argmax().item()),
            "fgsm_prediction": int(raw_pred[idx].item()),
            "fgsm_squeezed_prediction": int(binary_probs[idx].argmax().item()),
            "clean_l1_score": clean_score,
            "fgsm_l1_score": score,
            "threshold": threshold,
            "flagged_as_adversarial": bool(score > threshold),
        }
        (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(metrics.items())
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Visual: {RESULTS_DIR / 'mnist_fgsm_feature_squeezing.png'}", flush=True)


if __name__ == "__main__":
    main()
