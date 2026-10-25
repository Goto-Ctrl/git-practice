"""Train and evaluate a flow-level MLP using capture20110810.binetflow."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BOTNET_PREFIX = "flow=From-Botnet-"
BENIGN_PREFIXES = ("flow=From-Normal-", "flow=Normal-", "flow=Background-")
EXCLUDED_PREFIXES = ("flow=To-Background-", "flow=To-Normal-")


def target_from_label(label: str) -> str | None:
    """Return benign, an attack subtype, or None for excluded labels."""
    label = str(label).strip()
    if label.startswith(BOTNET_PREFIX):
        subtype = label[len(BOTNET_PREFIX):]
        return subtype or "unknown-botnet"
    if label.startswith(BENIGN_PREFIXES):
        return "benign"
    if label.startswith(EXCLUDED_PREFIXES):
        return None
    return None


def load_data(path: Path, max_rows: int | None, min_class_count: int) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=max_rows)
    required = {
        "StartTime", "Dur", "Proto", "Sport", "Dir", "Dport", "State",
        "TotPkts", "TotBytes", "SrcBytes", "Label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    frame["target"] = frame["Label"].map(target_from_label)
    frame = frame.dropna(subset=["target"]).copy()
    counts = frame["target"].value_counts()
    keep = counts[(counts >= min_class_count) | (counts.index == "benign")].index
    frame = frame[frame["target"].isin(keep)].copy()
    frame["StartTime"] = pd.to_datetime(frame["StartTime"], errors="coerce")
    for column in ("Dur", "Sport", "Dport", "TotPkts", "TotBytes", "SrcBytes"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("Proto", "Dir", "State"):
        frame[column] = frame[column].astype("string")
    frame = frame.dropna(
        subset=["StartTime", "Dur", "TotPkts", "TotBytes", "SrcBytes", "target"]
    )
    if frame["target"].nunique() < 2:
        raise ValueError("at least two target classes are required")
    return frame.sort_values("StartTime", kind="stable").reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build features without SrcAddr, DstAddr, or any endpoint identity."""
    features = frame[
        [
            "Dur", "Sport", "Dport", "TotPkts", "TotBytes", "SrcBytes",
            "Proto", "Dir", "State",
        ]
    ].copy()
    features["DstBytes"] = (features["TotBytes"] - features["SrcBytes"]).clip(lower=0)
    features["SrcByteRatio"] = (
        features["SrcBytes"] / features["TotBytes"].replace(0, np.nan)
    )
    for column in ("Dur", "TotPkts", "TotBytes", "SrcBytes", "DstBytes"):
        features[f"Log_{column}"] = np.log1p(features[column].clip(lower=0))
    numeric = [
        "Dur", "Sport", "Dport", "TotPkts", "TotBytes", "SrcBytes",
        "DstBytes", "SrcByteRatio", "Log_Dur", "Log_TotPkts",
        "Log_TotBytes", "Log_SrcBytes", "Log_DstBytes",
    ]
    categorical = ["Proto", "Dir", "State"]
    return features, numeric, categorical


def make_transformer(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return transform


class TorchFlowClassifier:
    """CPU/GPU-capable MLP with inverse-frequency class-weighted loss."""

    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        seed: int,
        epochs: int = 30,
        batch_size: int = 4096,
    ) -> None:
        self.transformer = make_transformer(numeric, categorical)
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_: np.ndarray | None = None
        self.network: nn.Module | None = None

    def _array(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.transformer.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=np.float32).copy()

    def fit(
        self,
        frame: pd.DataFrame,
        target: pd.Series,
        class_weights: np.ndarray | None = None,
    ) -> "TorchFlowClassifier":
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        features = self.transformer.fit_transform(frame)
        if hasattr(features, "toarray"):
            features = features.toarray()
        features = np.asarray(features, dtype=np.float32).copy()
        labels = target.to_numpy(dtype=np.int64).copy()
        self.classes_ = np.unique(labels)
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            raise ValueError("training labels must be contiguous integers starting at zero")

        if class_weights is None:
            counts = np.bincount(labels, minlength=len(self.classes_)).astype(np.float32)
            weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
        else:
            weights = np.asarray(class_weights, dtype=np.float32)
            if weights.shape != (len(self.classes_),):
                raise ValueError("class_weights must contain one value per training class")
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("class_weights must contain finite positive values")
        self.network = nn.Sequential(
            nn.Linear(features.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.classes_)),
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-3, weight_decay=1e-4)
        dataset = TensorDataset(
            torch.from_numpy(features), torch.from_numpy(labels)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=self.device.type == "cuda",
        )
        self.network.train()
        for _ in range(self.epochs):
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network(batch_features), batch_labels)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("model has not been fitted")
        features = torch.from_numpy(self._array(frame)).to(self.device)
        self.network.eval()
        with torch.inference_mode():
            probabilities = torch.softmax(self.network(features), dim=1)
        return probabilities.cpu().numpy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(frame).argmax(axis=1).astype(np.int64)


def split_by_time(
    frame: pd.DataFrame, train_fraction: float, validation_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(frame) * train_fraction)
    validation_end = train_end + int(len(frame) * validation_fraction)
    if train_end == 0 or validation_end >= len(frame):
        raise ValueError("time split produced an empty partition")
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def sample_training_data(
    frame: pd.DataFrame,
    samples_per_class: int,
    seed: int,
    max_oversample_factor: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Soften imbalance with bounded class-wise sampling on training data only."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if max_oversample_factor < 1:
        raise ValueError("max_oversample_factor must be positive")

    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []
    stats: dict[str, dict[str, int]] = {}
    for class_name, group in frame.groupby("target", sort=True):
        original_count = len(group)
        target_count = min(
            samples_per_class,
            max(original_count, original_count * max_oversample_factor),
        )
        replace = original_count < target_count
        indices = rng.choice(original_count, size=target_count, replace=replace)
        sampled = group.iloc[indices].copy()
        sampled_parts.append(sampled)
        stats[str(class_name)] = {
            "original": original_count,
            "sampled": len(sampled),
            "replacement": int(replace),
            "oversample_factor": float(len(sampled) / original_count),
        }

    balanced = pd.concat(sampled_parts, ignore_index=True)
    permutation = rng.permutation(len(balanced))
    return balanced.iloc[permutation].reset_index(drop=True), stats


def _binary_metrics(
    truth: np.ndarray, prediction: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    labels = np.array([0, 1], dtype=np.int64)
    one_hot = np.eye(2, dtype=np.int64)[truth]
    return {
        "rows": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_macro": float(
            precision_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "attack_precision": float(precision_score(truth, prediction, pos_label=1, zero_division=0)),
        "attack_recall": float(recall_score(truth, prediction, pos_label=1, zero_division=0)),
        "attack_f1": float(f1_score(truth, prediction, pos_label=1, zero_division=0)),
        "pr_auc_attack": float(average_precision_score(truth, probabilities[:, 1])),
        "classification_report": classification_report(
            truth, prediction, labels=labels, target_names=["benign", "attack"],
            output_dict=True, zero_division=0
        ),
        "classes": ["benign", "attack"],
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
    }


def choose_attack_threshold(
    truth: np.ndarray, attack_probability: np.ndarray
) -> float:
    """Choose a validation-only threshold maximizing attack F1, then accuracy."""
    best_score = (-1.0, -1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 19):
        prediction = (attack_probability >= threshold).astype(np.int64)
        score = (
            f1_score(truth, prediction, pos_label=1, zero_division=0),
            accuracy_score(truth, prediction),
            -threshold,
        )
        if score > best_score:
            best_score = (score[0], score[1], threshold)
    return float(best_score[2])


def evaluate_stage_one(
    model: TorchFlowClassifier, frame: pd.DataFrame, threshold: float
) -> dict[str, object]:
    features, _, _ = make_features(frame)
    truth = (frame["target"] != "benign").to_numpy(dtype=np.int64)
    probabilities = model.predict_proba(features)
    prediction = (probabilities[:, 1] >= threshold).astype(np.int64)
    result = _binary_metrics(truth, prediction, probabilities)
    result["threshold"] = threshold
    return result


def evaluate_stage_two(
    model: TorchFlowClassifier,
    frame: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> dict[str, object]:
    known_classes = set(label_encoder.classes_)
    attacks = frame[
        frame["target"].isin(known_classes) & (frame["target"] != "benign")
    ].copy()
    excluded = frame[
        (frame["target"] != "benign") & ~frame["target"].isin(known_classes)
    ]["target"].value_counts().to_dict()
    if attacks.empty:
        return {
            "rows": 0,
            "excluded_unknown_class_rows": int(sum(excluded.values())),
            "excluded_unknown_classes": excluded,
            "classes": label_encoder.classes_.tolist(),
            "confusion_matrix": [],
        }
    features, _, _ = make_features(attacks)
    truth = label_encoder.transform(attacks["target"])
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    class_ids = np.arange(len(label_encoder.classes_), dtype=np.int64)
    one_hot = np.eye(len(class_ids), dtype=np.int64)[truth]
    return {
        "rows": int(len(attacks)),
        "excluded_unknown_class_rows": int(sum(excluded.values())),
        "excluded_unknown_classes": excluded,
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_macro": float(
            precision_score(truth, prediction, labels=class_ids, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(truth, prediction, labels=class_ids, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(truth, prediction, labels=class_ids, average="macro", zero_division=0)
        ),
        "pr_auc_macro_ovr": float(
            average_precision_score(one_hot, probabilities, average="macro")
        ),
        "classification_report": classification_report(
            truth, prediction, labels=class_ids,
            target_names=label_encoder.classes_, output_dict=True, zero_division=0
        ),
        "classes": label_encoder.classes_.tolist(),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=class_ids
        ).tolist(),
    }


def evaluate_cascade(
    stage_one: TorchFlowClassifier,
    stage_two: TorchFlowClassifier,
    frame: pd.DataFrame,
    attack_encoder: LabelEncoder,
    threshold: float,
) -> dict[str, object]:
    features, _, _ = make_features(frame)
    stage_one_probability = stage_one.predict_proba(features)[:, 1]
    stage_one_prediction = stage_one_probability >= threshold
    final = np.where(stage_one_prediction, "attack", "benign").astype(object)
    attack_rows = np.flatnonzero(stage_one_prediction)
    if len(attack_rows):
        attack_prediction = attack_encoder.inverse_transform(
            stage_two.predict(features.iloc[attack_rows])
        )
        final[attack_rows] = np.array([f"attack:{name}" for name in attack_prediction])
    truth = np.where(
        frame["target"].to_numpy() == "benign",
        "benign",
        np.array([f"attack:{name}" for name in frame["target"]]),
    )
    labels = sorted(set(truth) | set(final))
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(truth, final)),
        "macro_f1": float(f1_score(truth, final, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            truth, final, labels=labels, output_dict=True, zero_division=0
        ),
        "classes": labels,
        "confusion_matrix": confusion_matrix(
            truth, final, labels=labels
        ).tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binetflow",
        type=Path,
        default=Path(__file__).with_name("capture20110810.binetflow.txt"),
        help="capture20110810.binetflow CSV export",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("binetflow_results"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument(
        "--train-samples-per-class",
        type=int,
        default=10000,
        help="number of training rows per class after time split (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if not args.binetflow.is_file():
        parser.error(f"file not found: {args.binetflow}")
    if args.min_class_count < 1:
        parser.error("--min-class-count must be positive")
    if args.train_samples_per_class < 1:
        parser.error("--train-samples-per-class must be positive")

    data = load_data(args.binetflow, args.max_rows, args.min_class_count)
    train, validation, test = split_by_time(data, 0.70, 0.15)
    train["stage_one_target"] = (train["target"] != "benign").astype(np.int64)
    stage_one_counts = np.bincount(
        train["stage_one_target"].to_numpy(), minlength=2
    ).astype(np.float32)
    stage_one_raw_weights = np.sqrt(
        stage_one_counts.sum() / (2 * np.maximum(stage_one_counts, 1.0))
    )
    stage_one_weights = np.clip(
        stage_one_raw_weights / np.mean(stage_one_raw_weights), 0.25, 4.0
    )
    stage_one_train, stage_one_sampling = sample_training_data(
        train.assign(target=np.where(train["stage_one_target"] == 1, "attack", "benign")),
        args.train_samples_per_class,
        args.seed,
    )
    stage_one_train_features, numeric, categorical = make_features(stage_one_train)
    stage_one_model = TorchFlowClassifier(numeric, categorical, args.seed)
    stage_one_model.fit(
        stage_one_train_features,
        stage_one_train["stage_one_target"],
        class_weights=stage_one_weights,
    )

    attack_train = train[train["target"] != "benign"].copy()
    if attack_train.empty:
        raise ValueError("training partition contains no attack rows")
    attack_encoder = LabelEncoder()
    attack_train["target_id"] = attack_encoder.fit_transform(attack_train["target"])
    attack_counts = np.bincount(
        attack_train["target_id"].to_numpy(dtype=np.int64),
        minlength=len(attack_encoder.classes_),
    ).astype(np.float32)
    attack_raw_weights = np.sqrt(
        attack_counts.sum()
        / (len(attack_counts) * np.maximum(attack_counts, 1.0))
    )
    attack_weights = np.clip(
        attack_raw_weights / np.mean(attack_raw_weights), 0.25, 4.0
    )
    attack_train_sampled, attack_sampling = sample_training_data(
        attack_train, args.train_samples_per_class, args.seed
    )
    attack_features, attack_numeric, attack_categorical = make_features(
        attack_train_sampled
    )
    stage_two_model = TorchFlowClassifier(
        attack_numeric, attack_categorical, args.seed + 1
    )
    stage_two_model.fit(
        attack_features,
        attack_train_sampled["target_id"],
        class_weights=attack_weights,
    )

    stage_one_validation_truth = (
        validation["target"] != "benign"
    ).to_numpy(dtype=np.int64)
    validation_features, _, _ = make_features(validation)
    validation_probability = stage_one_model.predict_proba(validation_features)[:, 1]
    threshold = choose_attack_threshold(
        stage_one_validation_truth, validation_probability
    )

    result = {
        "input": str(args.binetflow),
        "seed": args.seed,
        "classification": {
            "architecture": "stage_one_benign_vs_attack_then_stage_two_attack_type",
            "stage_one_threshold": threshold,
            "validation_only_threshold_selection": True,
        },
        "stage_one": {
            "sampling": stage_one_sampling,
            "class_weights": {
                label: {
                    "original_training_rows": int(stage_one_counts[index]),
                    "raw_sqrt_weight": float(stage_one_raw_weights[index]),
                    "weight": float(stage_one_weights[index]),
                }
                for index, label in enumerate(["benign", "attack"])
            },
        },
        "stage_two": {
            "sampling": attack_sampling,
            "class_weights": {
                label: {
                    "original_training_rows": int(attack_counts[index]),
                    "raw_sqrt_weight": float(attack_raw_weights[index]),
                    "weight": float(attack_weights[index]),
                }
                for index, label in enumerate(attack_encoder.classes_)
            },
        },
        "label_policy": {
            "From-Botnet-*": "attack subtype after From-Botnet-",
            "From-Normal-*, Normal-*, Background-*": "benign",
            "To-Background-* and To-Normal-*": "excluded",
            "rare classes": f"excluded below {args.min_class_count} rows",
        },
        "excluded_features": ["SrcAddr", "DstAddr"],
        "partitions": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "time_ranges": {
                name: [str(part["StartTime"].min()), str(part["StartTime"].max())]
                for name, part in (
                    ("train", train), ("validation", validation), ("test", test)
                )
            },
        },
        "label_mapping": {
            "stage_one": {"0": "benign", "1": "attack"},
            "stage_two": {
                str(index): label
                for index, label in enumerate(attack_encoder.classes_)
            },
        },
        "validation": {
            "stage_one": evaluate_stage_one(
                stage_one_model, validation, threshold
            ),
            "stage_two": evaluate_stage_two(
                stage_two_model, validation, attack_encoder
            ),
            "cascade": evaluate_cascade(
                stage_one_model,
                stage_two_model,
                validation,
                attack_encoder,
                threshold,
            ),
        },
        "test": {
            "stage_one": evaluate_stage_one(stage_one_model, test, threshold),
            "stage_two": evaluate_stage_two(
                stage_two_model, test, attack_encoder
            ),
            "cascade": evaluate_cascade(
                stage_one_model,
                stage_two_model,
                test,
                attack_encoder,
                threshold,
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test_report = result["test"]["cascade"]
    pd.DataFrame(
        test_report["confusion_matrix"],
        index=test_report["classes"],
        columns=test_report["classes"],
    ).to_csv(args.output_dir / "confusion_matrix.csv", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
