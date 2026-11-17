"""
ASVspoof 2019 Logical Access (LA) PyTorch Dataset and DataLoader.

Reads directly from the extracted challenge directory at:
C:\\Users\\manoj\\Downloads\\archive (4)\\LA\\LA

Key features:
- Reads CM (Countermeasure) protocol files strictly (25,380 train, 24,844 dev, 71,237 eval).
- Ignores all extra ASV enrollment files in the directory.
- Maps labels strictly: spoof = 0, bonafide = 1.
- Implements variable-length dynamic batching with zero-padding and true length tracking.
- Does NOT blindly repeat-pad waveforms, avoiding artificial periodicity artifacts.
- Zero copying or moving of the 24 GB dataset.
"""

import os
from typing import Optional, Tuple, Dict, Any, List
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_DATA_ROOT = r"C:\Users\manoj\Downloads\archive (4)\LA\LA"


class ASVSpoof2019LADataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 Logical Access (LA) track.

    Args:
        data_root: Root directory of ASVspoof 2019 LA track (defaults to extracted archive).
        split: Partition to load ('train', 'dev', or 'eval').
        max_duration_sec: Maximum audio duration in seconds (default: 4.0s = 64,000 samples @ 16kHz).
        is_train: If True, randomly offsets audio exceeding max_duration_sec as data augmentation.
        limit: Optional maximum number of items to load (useful for smoke tests and unit testing).
    """

    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        split: str = "train",
        max_duration_sec: float = 4.0,
        is_train: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.split = split.lower().strip()
        self.max_duration_sec = max_duration_sec
        self.target_sr = 16000
        self.max_samples = int(self.target_sr * max_duration_sec)  # 64,000 samples
        self.is_train = is_train

        valid_splits = {"train", "dev", "eval"}
        if self.split not in valid_splits:
            raise ValueError(f"Unknown split '{self.split}'. Must be one of {valid_splits}.")

        # Determine protocol and flac directory paths
        protocol_dir = os.path.join(self.data_root, "ASVspoof2019_LA_cm_protocols")
        if self.split == "train":
            protocol_file = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.train.trn.txt")
            self.audio_dir = os.path.join(self.data_root, "ASVspoof2019_LA_train", "flac")
        elif self.split == "dev":
            protocol_file = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
            self.audio_dir = os.path.join(self.data_root, "ASVspoof2019_LA_dev", "flac")
        else:  # eval
            protocol_file = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.eval.trl.txt")
            self.audio_dir = os.path.join(self.data_root, "ASVspoof2019_LA_eval", "flac")

        if not os.path.isfile(protocol_file):
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        # Parse protocol file
        self.entries: List[Dict[str, Any]] = []
        self.n_bonafide = 0
        self.n_spoof = 0

        with open(protocol_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                speaker_id = parts[0]
                file_id = parts[1]
                system_id = parts[2]
                attack_id = parts[3]
                key_label = parts[4].lower()

                # Label mapping: spoof = 0, bonafide = 1
                if key_label == "bonafide":
                    label = 1
                    self.n_bonafide += 1
                elif key_label == "spoof":
                    label = 0
                    self.n_spoof += 1
                else:
                    continue

                audio_path = os.path.join(self.audio_dir, f"{file_id}.flac")
                self.entries.append({
                    "speaker_id": speaker_id,
                    "file_id": file_id,
                    "system_id": system_id,
                    "attack_id": attack_id,
                    "key_label": key_label,
                    "label": label,
                    "audio_path": audio_path,
                })

                if limit is not None and len(self.entries) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, int, str, str]:
        """
        Loads and returns single audio recording and metadata.

        Returns:
            Tuple of:
            - audio: 1D float32 numpy array
            - label: int (0 for spoof, 1 for bonafide)
            - length: int (true audio sample length <= max_samples)
            - file_id: str
            - attack_id: str
        """
        entry = self.entries[idx]
        audio_path = entry["audio_path"]
        label = entry["label"]

        # Read audio file using soundfile
        audio, sr = sf.read(audio_path, dtype="float32")

        # Convert to mono if multi-channel (ASVspoof is natively mono)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        total_samples = len(audio)

        # Handle duration truncation / random windowing
        if total_samples > self.max_samples:
            if self.is_train:
                # Random window during training for temporal augmentation
                max_offset = total_samples - self.max_samples
                start = np.random.randint(0, max_offset + 1)
                audio = audio[start : start + self.max_samples]
            else:
                # Deterministic first window for dev / evaluation
                audio = audio[: self.max_samples]
            true_length = self.max_samples
        else:
            true_length = total_samples

        return audio, label, true_length, entry["file_id"], entry["attack_id"]

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes balanced inverse-frequency class weights based on actual partition counts.

        Returns:
            torch.FloatTensor of shape [2] where [0] is spoof weight, [1] is bonafide weight.
        """
        total = self.n_bonafide + self.n_spoof
        if self.n_spoof == 0 or self.n_bonafide == 0:
            return torch.tensor([1.0, 1.0], dtype=torch.float32)

        w_spoof = total / (2.0 * self.n_spoof)
        w_bonafide = total / (2.0 * self.n_bonafide)
        return torch.tensor([w_spoof, w_bonafide], dtype=torch.float32)


def collate_variable_audio(
    batch: List[Tuple[np.ndarray, int, int, str, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[str]]]:
    """
    Variable-length batch collator with zero-padding and length tracking.

    Avoids repeat-padding artifacts by padding with zeros to batch maximum length.

    Args:
        batch: List of tuples from ASVSpoof2019LADataset.__getitem__.

    Returns:
        Tuple of:
        - padded_audio: FloatTensor of shape [Batch, max_len_in_batch]
        - labels: LongTensor of shape [Batch] (0 = spoof, 1 = bonafide)
        - lengths: LongTensor of shape [Batch] (valid sample lengths)
        - metadata: Dict with 'file_ids' and 'attack_ids' lists
    """
    batch_size = len(batch)
    max_len = max(len(item[0]) for item in batch)

    # Allocate zero-padded batch tensor
    padded_audio = torch.zeros((batch_size, max_len), dtype=torch.float32)
    labels = torch.empty(batch_size, dtype=torch.long)
    lengths = torch.empty(batch_size, dtype=torch.long)
    file_ids: List[str] = []
    attack_ids: List[str] = []

    for i, (audio, label, length, file_id, attack_id) in enumerate(batch):
        padded_audio[i, : len(audio)] = torch.from_numpy(audio)
        labels[i] = label
        lengths[i] = length
        file_ids.append(file_id)
        attack_ids.append(attack_id)

    metadata = {
        "file_ids": file_ids,
        "attack_ids": attack_ids,
    }
    return padded_audio, labels, lengths, metadata


def create_asvspoof_dataloader(
    data_root: str = DEFAULT_DATA_ROOT,
    split: str = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    max_duration_sec: float = 4.0,
    limit: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Factory creating a DataLoader for ASVspoof 2019 LA partition.

    Args:
        data_root: Extracted LA directory root.
        split: 'train', 'dev', or 'eval'.
        batch_size: Batch size (default: 32).
        shuffle: If None, defaults to True for train, False for dev/eval.
        num_workers: DataLoader subprocesses (0 for Windows stability).
        max_duration_sec: Duration cutoff in seconds (default: 4.0s).
        limit: Optional cap on loaded items.
        pin_memory: Pin memory for fast GPU transfer.

    Returns:
        Configured PyTorch DataLoader.
    """
    is_train = (split == "train")
    if shuffle is None:
        shuffle = is_train

    dataset = ASVSpoof2019LADataset(
        data_root=data_root,
        split=split,
        max_duration_sec=max_duration_sec,
        is_train=is_train,
        limit=limit,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_audio,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=is_train,
    )
    return loader
