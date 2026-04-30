"""
Download and convert the NSL-KDD dataset used by the CLAPP paper.

Outputs:
    data/NSL-KDD-train.csv
    data/NSL-KDD-test.csv
    data/NSL-KDD-full.csv

The CSV keeps all 41 NSL-KDD features plus:
    label      -> 0 Normal, 1 DoS, 2 Probe, 3 R2L, 4 U2R
    label_name -> original attack type, useful for inspection
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd


FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

SOURCES = {
    "train": [
        (
            "KDDTrain+.txt",
            "https://zenodo.org/records/3768048/files/KDDTrain%2B.txt?download=1",
        ),
        (
            "KDDTrain+.arff",
            "https://zenodo.org/records/3768048/files/KDDTrain%2B.arff?download=1",
        ),
        (
            "KDDTrain+.arff",
            "https://www.unb.ca/cic/datasets/nsl-kdd/KDDTrain+.arff",
        ),
    ],
    "test": [
        (
            "KDDTest+.txt",
            "https://zenodo.org/records/3768048/files/KDDTest%2B.txt?download=1",
        ),
        (
            "KDDTest+.arff",
            "https://zenodo.org/records/3768048/files/KDDTest%2B.arff?download=1",
        ),
        (
            "KDDTest+.arff",
            "https://www.unb.ca/cic/datasets/nsl-kdd/KDDTest+.arff",
        ),
    ],
}

ATTACK_FAMILY = {
    # Normal
    "normal": 0,
    # DoS
    "apache2": 1,
    "back": 1,
    "land": 1,
    "mailbomb": 1,
    "neptune": 1,
    "pod": 1,
    "processtable": 1,
    "smurf": 1,
    "teardrop": 1,
    "udpstorm": 1,
    "worm": 1,
    # Probe
    "ipsweep": 2,
    "mscan": 2,
    "nmap": 2,
    "portsweep": 2,
    "saint": 2,
    "satan": 2,
    # R2L
    "ftp_write": 3,
    "guess_passwd": 3,
    "httptunnel": 3,
    "imap": 3,
    "multihop": 3,
    "named": 3,
    "phf": 3,
    "sendmail": 3,
    "snmpgetattack": 3,
    "snmpguess": 3,
    "spy": 3,
    "warezclient": 3,
    "warezmaster": 3,
    "xlock": 3,
    "xsnoop": 3,
    # U2R
    "buffer_overflow": 4,
    "loadmodule": 4,
    "perl": 4,
    "ps": 4,
    "rootkit": 4,
    "sqlattack": 4,
    "xterm": 4,
}


def _clean_cell(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _iter_arff_rows(arff_path: Path) -> Iterable[list[str]]:
    in_data = False
    with arff_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data:
                continue
            for row in csv.reader([line]):
                yield [_clean_cell(cell) for cell in row]


def parse_arff_to_pandas(arff_path: str | Path) -> pd.DataFrame:
    rows = []
    expected_cols = len(FEATURE_NAMES) + 2
    for row in _iter_arff_rows(Path(arff_path)):
        if len(row) != expected_cols:
            raise ValueError(
                f"Unexpected ARFF row width in {arff_path}: got {len(row)}, expected {expected_cols}"
            )
        rows.append(row)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["attack_type", "difficulty"])
    df["attack_type"] = df["attack_type"].str.lower().str.rstrip(".")
    return df


def parse_txt_to_pandas(txt_path: str | Path) -> pd.DataFrame:
    rows = []
    expected_cols = len(FEATURE_NAMES) + 2
    with Path(txt_path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) != expected_cols:
                raise ValueError(
                    f"Unexpected TXT row width in {txt_path}: got {len(row)}, expected {expected_cols}"
                )
            rows.append([_clean_cell(cell) for cell in row])

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["attack_type", "difficulty"])
    df["attack_type"] = df["attack_type"].str.lower().str.rstrip(".")
    return df


def parse_dataset_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".arff":
        return parse_arff_to_pandas(path)
    return parse_txt_to_pandas(path)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (NSL-KDD downloader)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _get_source(split: str, output_dir: Path) -> Path:
    for filename, _ in SOURCES[split]:
        local_path = output_dir / filename
        if local_path.exists():
            print(f"[download] Using local {local_path}")
            return local_path

    print(f"[download] Fetching {split} set from NSL-KDD mirror...")
    last_error = None
    for filename, url in SOURCES[split]:
        local_path = output_dir / filename
        try:
            _download(url, local_path)
            return local_path
        except Exception as exc:
            last_error = exc
            if local_path.exists():
                local_path.unlink()
    raise RuntimeError(f"all download sources failed for {split}: {last_error}")


def _encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(df["attack_type"]) - set(ATTACK_FAMILY))
    if unknown:
        raise ValueError(f"Unknown NSL-KDD attack types: {unknown}")

    encoded = df.copy()
    encoded["label_name"] = encoded["attack_type"]
    encoded["label"] = encoded["attack_type"].map(ATTACK_FAMILY).astype(int)

    numeric_cols = [c for c in FEATURE_NAMES if c not in {"protocol_type", "service", "flag"}]
    for col in numeric_cols:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce").fillna(0)

    return encoded[FEATURE_NAMES + ["label", "label_name"]]


def download_nslkdd(output_dir: str = "data", train_size: int | None = None) -> str | None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for split in ("train", "test"):
        try:
            source_path = _get_source(split, out_dir)
            print("[parse] Converting source file to DataFrame...")
            df = parse_dataset_file(source_path)
            df = _encode_labels(df)

            if split == "train" and train_size is not None:
                df = df.iloc[:train_size].copy()
                print(f"[limit] Keeping first {train_size} train samples.")

            csv_path = out_dir / f"NSL-KDD-{split}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(df)} rows -> {csv_path}")
            outputs.append(df)
        except Exception as exc:
            print(f"Failed to prepare {split} split: {exc}")
            print(f"Manual sources: {[url for _, url in SOURCES[split]]}")
            return None

    full = pd.concat(outputs, ignore_index=True)
    full_path = out_dir / "NSL-KDD-full.csv"
    full.to_csv(full_path, index=False)
    print(f"Combined dataset -> {full_path} ({len(full)} rows)")
    return str(full_path)


def verify_dataset(csv_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    print()
    print(f"Dataset: {csv_path}")
    print(f"Shape: {df.shape}")
    print()
    print("Attack type distribution:")
    print(df["label_name"].value_counts().to_string())
    print()
    print("Family label distribution:")
    print(df["label"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NSL-KDD dataset")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--train_size", type=int, default=None, help="Limit train split for testing")
    args = parser.parse_args()

    csv_path = download_nslkdd(args.output, args.train_size)
    if csv_path:
        verify_dataset(csv_path)


if __name__ == "__main__":
    main()
