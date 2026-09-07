import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import torch_npu
except ImportError:
    torch_npu = None

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))
from model import ModelA

FIELDS = ("text_clue_bert", "audio_clue_bert", "visual_clue_bert", "final_description_bert")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default=str(PACKAGE / "pkl/data_a.pkl"))
    p.add_argument("--bert_dir", default=str(PACKAGE / "bert-base-uncased"))
    p.add_argument("--output_dir", default=str(PACKAGE / "outputs"))
    p.add_argument("--run_name", default="run_a")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=160)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--feature_layers", type=int, default=2)
    p.add_argument("--align_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--scale_e", type=float, default=0.15)
    p.add_argument("--mode_e", choices=("a", "b", "none"), default="b")
    p.add_argument("--limit_x", type=float, default=0.1)
    p.add_argument("--scale_x", type=float, default=2.0)
    p.add_argument("--flag_i", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mode_i", choices=("i", "h", "none"), default="i")
    p.add_argument("--width_i", type=int, default=3)
    p.add_argument("--scale_i", type=float, default=0.08)
    p.add_argument("--offset_i", type=float, default=0.0)
    p.add_argument("--gain_i", type=float, default=0.20)
    p.add_argument("--flag_h", action="store_true")
    p.add_argument("--bound_h", type=float, default=0.35)
    p.add_argument("--sharp_h", type=float, default=8.0)
    p.add_argument("--amp_h", type=float, default=1.0)
    p.add_argument("--flag_j", action="store_true")
    p.add_argument("--scale_j", type=float, default=0.0)
    p.add_argument("--scale_k", type=float, default=0.0)
    p.add_argument("--has0_sign_bce_weight", type=float, default=0.125)
    p.add_argument("--has0_sign_temperature", type=float, default=0.35)
    p.add_argument("--has0_exact_zero_boost", type=float, default=2.0)
    p.add_argument("--zero_nonneg_hinge_weight", type=float, default=0.025)
    p.add_argument("--zero_nonneg_margin", type=float, default=0.05)
    p.add_argument("--loss_type", choices=("mse", "smooth_l1"), default="smooth_l1")
    p.add_argument("--smooth_l1_beta", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.997)
    p.add_argument("--ema_start_epoch", type=int, default=4)
    p.add_argument("--eval_ema", action="store_true", default=True)
    p.add_argument("--selection_metric", default="loss")
    p.add_argument("--selection_mode", choices=("max", "min"), default="min")
    p.add_argument("--composite_corr_weight", type=float, default=1.0)
    p.add_argument("--composite_mae_weight", type=float, default=0.50)
    p.add_argument("--composite_mult5_weight", type=float, default=0.20)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261100)
    p.add_argument("--torch_threads", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)
    return p.parse_args()


def device(name):
    if name == "auto":
        if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_device(0)
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name.startswith("npu"):
        if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU is unavailable")
        index = int(name.split(":", 1)[1]) if ":" in name else 0
        torch.npu.set_device(index)
        return torch.device(f"npu:{index}")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return torch.device(name)
    return torch.device("cpu")


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def metrics(pred, truth):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    mae = float(np.mean(np.abs(pred - truth)))
    corr = float(np.corrcoef(pred, truth)[0, 1]) if len(truth) > 1 and np.std(pred) > 0 and np.std(truth) > 0 else 0.0
    has0 = truth >= 0
    pred_has0 = pred >= 0
    nonzero = truth != 0
    if np.any(nonzero):
        nz_truth, nz_pred = truth[nonzero] > 0, pred[nonzero] > 0
        nonzero_acc = float(accuracy_score(nz_truth, nz_pred))
        nonzero_f1 = float(f1_score(nz_truth, nz_pred, average="weighted", zero_division=0))
    else:
        nonzero_acc = nonzero_f1 = 0.0
    rounded = lambda x, limit: np.round(np.clip(x, -limit, limit))
    exact_zero = truth == 0
    pred_zero = np.round(pred) == 0
    true_zero = np.round(truth) == 0
    tp = float(np.sum(pred_zero & true_zero))
    fp = float(np.sum(pred_zero & ~true_zero))
    fn = float(np.sum(~pred_zero & true_zero))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    zero_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "Has0_acc_2": round(float(accuracy_score(has0, pred_has0)), 4),
        "Has0_F1_score": round(float(f1_score(has0, pred_has0, average="weighted", zero_division=0)), 4),
        "Non0_acc_2": round(nonzero_acc, 4),
        "Non0_F1_score": round(nonzero_f1, 4),
        "Mult_acc_3": round(float(np.mean(rounded(pred, 1) == rounded(truth, 1))), 4),
        "Mult_acc_5": round(float(np.mean(rounded(pred, 2) == rounded(truth, 2))), 4),
        "Mult_acc_7": round(float(np.mean(rounded(pred, 3) == rounded(truth, 3))), 4),
        "MAE": round(mae, 4),
        "Corr": round(corr, 4),
        "Zero_recall": round(recall, 4),
        "Zero_precision": round(precision, 4),
        "Zero_F1": round(zero_f1, 4),
        "Zero_pred_rate": round(float(np.mean(pred_zero)), 4),
        "ExactZero_nonneg_acc": round(float(np.mean(pred[exact_zero] >= 0)) if np.any(exact_zero) else 0.0, 4),
        "ExactZero_count": int(np.sum(exact_zero)),
    }


class Data(Dataset):
    def __init__(self, split, limit=0):
        self.data = split
        self.size = len(split["regression_labels"]) if not limit else min(len(split["regression_labels"]), limit)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        item = {
            "text": torch.as_tensor(self.data["text"][i], dtype=torch.float32),
            "audio": torch.as_tensor(self.data["audio"][i], dtype=torch.float32),
            "vision": torch.as_tensor(self.data["vision"][i], dtype=torch.float32),
            "u_g": torch.as_tensor(self.data["fusion" + "_weights"][i], dtype=torch.float32),
            "regression_labels": torch.tensor(float(self.data["regression_labels"][i]), dtype=torch.float32),
        }
        for field in FIELDS:
            item[field] = torch.as_tensor(self.data[field][i], dtype=torch.long)
        return item


def move(batch, target):
    return {k: v.to(target, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def loss_fn(output, labels, criterion, model):
    prediction = output["prediction"]
    loss = criterion(prediction, labels)
    if model.has0_sign_bce_weight > 0:
        target = (labels >= 0).to(prediction.dtype)
        bce = F.binary_cross_entropy_with_logits(prediction / max(model.has0_sign_temperature, 1e-6), target, reduction="none")
        zero = labels == 0
        if torch.any(zero):
            bce = bce * (1 + model.has0_exact_zero_boost * zero.to(bce.dtype))
        loss = loss + model.has0_sign_bce_weight * bce.mean()
    if model.zero_nonneg_hinge_weight > 0:
        zero = labels == 0
        if torch.any(zero):
            loss = loss + model.zero_nonneg_hinge_weight * F.relu(model.zero_nonneg_margin - prediction[zero]).mean()
    if model.scale_j > 0 and "u_j" in output:
        loss = loss + model.scale_j * torch.stack([criterion(x, labels) for x in output["u_j"].values()]).mean()
    return loss


def run_epoch(model, loader, criterion, target, optimizer=None, ema=None, decay=0.0, clip=1.0):
    training = optimizer is not None
    model.train(training)
    total, count, pred, truth = 0.0, 0, [], []
    for batch in tqdm(loader, leave=False):
        batch = move(batch, target)
        labels = batch["regression_labels"]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            loss = loss_fn(output, labels, criterion, model) if training else criterion(output["prediction"], labels)
            if training and model.scale_k > 0:
                reference_values = batch["u_g"]
                reference_values = reference_values / reference_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                coeff_x = output["u_h"].clamp_min(1e-6)
                loss = loss + model.scale_k * F.kl_div(coeff_x.log(), reference_values, reduction="batchmean")
            if training:
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                if ema is not None:
                    for key, value in model.state_dict().items():
                        if key in ema:
                            ema[key].mul_(decay).add_(value.detach(), alpha=1 - decay)
        n = labels.numel()
        total += float(loss.detach().cpu()) * n
        count += n
        pred.append(output["prediction"].detach().cpu().numpy())
        truth.append(labels.detach().cpu().numpy())
    return total / max(count, 1), metrics(np.concatenate(pred), np.concatenate(truth))


def swap_ema(model, ema):
    current = model.state_dict()
    old = {}
    for key, value in ema.items():
        if key in current:
            old[key] = current[key].detach().clone()
            current[key].copy_(value)
    return old


def restore(model, old):
    current = model.state_dict()
    for key, value in old.items():
        current[key].copy_(value)


def score(view, cfg):
    if cfg.selection_metric == "loss":
        return view["loss"]
    if cfg.selection_metric == "composite":
        m = view["metrics"]
        return cfg.composite_corr_weight * m["Corr"] - cfg.composite_mae_weight * m["MAE"] + cfg.composite_mult5_weight * m["Mult_acc_5"]
    return view["metrics"][cfg.selection_metric]


def better(value, best, mode):
    return best is None or (value > best if mode == "max" else value < best)


def main():
    cfg = args()
    seed(cfg.seed)
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    target = device(cfg.device)
    with Path(cfg.pkl).open("rb") as f:
        data = pickle.load(f)
    train_loader = DataLoader(Data(data["train"], cfg.max_train_samples), cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(Data(data["valid"], cfg.max_eval_samples), cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(Data(data["test"], cfg.max_eval_samples), cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)
    model = ModelA(
        bert_dir=cfg.bert_dir,
        text_dim=data["train"]["text"].shape[-1],
        audio_dim=data["train"]["audio"].shape[-1],
        vision_dim=data["train"]["vision"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        heads=cfg.heads,
        feature_layers=cfg.feature_layers,
        align_layers=cfg.align_layers,
        dropout=cfg.dropout,
        scale_a=cfg.scale_e,
        limit_x=cfg.limit_x,
        scale_x=cfg.scale_x,
        mode_e=cfg.mode_e,
        flag_i=cfg.flag_i,
        mode_i=cfg.mode_i,
        width_i=cfg.width_i,
        sigma_i=cfg.scale_i,
        offset_i=cfg.offset_i,
        scale_i=cfg.gain_i,
        flag_h=cfg.flag_h,
        bound_h=cfg.bound_h,
        sharp_h=cfg.sharp_h,
        amp_h=cfg.amp_h,
        flag_j=cfg.flag_j,
    ).to(target)
    model.scale_k = cfg.scale_k
    model.scale_j = cfg.scale_j
    model.has0_sign_bce_weight = cfg.has0_sign_bce_weight
    model.has0_sign_temperature = cfg.has0_sign_temperature
    model.has0_exact_zero_boost = cfg.has0_exact_zero_boost
    model.zero_nonneg_hinge_weight = cfg.zero_nonneg_hinge_weight
    model.zero_nonneg_margin = cfg.zero_nonneg_margin
    trainable = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.SmoothL1Loss(beta=cfg.smooth_l1_beta) if cfg.loss_type == "smooth_l1" else nn.MSELoss()
    ema = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("bert.") and torch.is_floating_point(v)} if cfg.use_ema else None
    out = Path(cfg.output_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    best_value, best_record, bad = None, None, 0
    log_path = out / "train_log.jsonl"
    for epoch in range(1, cfg.epochs + 1):
        epoch_ema = ema if ema is not None and epoch >= cfg.ema_start_epoch else None
        train_loss, valid_raw = run_epoch(model, train_loader, criterion, target, optimizer, epoch_ema, cfg.ema_decay, cfg.grad_clip)
        valid_loss, valid_metrics = run_epoch(model, valid_loader, criterion, target)
        test_loss, test_metrics = run_epoch(model, test_loader, criterion, target)
        selected = {"loss": valid_loss, "metrics": valid_metrics}
        selected_test = {"loss": test_loss, "metrics": test_metrics}
        source = "raw"
        if cfg.eval_ema and ema is not None and epoch >= cfg.ema_start_epoch:
            old = swap_ema(model, ema)
            try:
                selected_loss, selected_metrics = run_epoch(model, valid_loader, criterion, target)
                selected_test_loss, selected_test_metrics = run_epoch(model, test_loader, criterion, target)
                selected = {"loss": selected_loss, "metrics": selected_metrics}
                selected_test = {"loss": selected_test_loss, "metrics": selected_test_metrics}
                source = "ema"
            finally:
                restore(model, old)
        value = score(selected, cfg)
        record = {"epoch": epoch, "train_loss": train_loss, "valid": selected, "test": selected_test, "selection_source": source, "selection_score": value}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if better(value, best_value, cfg.selection_mode):
            best_value, best_record, bad = value, record, 0
            if source == "ema":
                old = swap_ema(model, ema)
                try:
                    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                finally:
                    restore(model, old)
            else:
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "args": vars(cfg), "record": record, "model_state": state}, out / "best.pt")
        else:
            bad += 1
        (out / "status.json").write_text(json.dumps({"epoch": epoch, "best_epoch": best_record["epoch"] if best_record else None, "record": record}, indent=2), encoding="utf-8")
        if cfg.early_stop_patience and bad >= cfg.early_stop_patience:
            break
    print(json.dumps(best_record, indent=2))


if __name__ == "__main__":
    main()
