from torch.utils.data import Dataset, DataLoader
import chess
import torch


class ChessDataset(Dataset):
    def __init__(self, positions, metadata, scores):
        self.positions = positions
        self.metadata = metadata
        self.scores = scores

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        board = torch.tensor(self.positions[idx], dtype=torch.float32).permute(2, 0, 1)

        meta = self.metadata[idx]
        move_number = meta["move_number"]
        castling_rights = [
            int(bool(meta["castling_rights"] & chess.BB_H1)),  # White kingside
            int(bool(meta["castling_rights"] & chess.BB_A1)),  # White queenside
            int(bool(meta["castling_rights"] & chess.BB_H8)),  # Black kingside
            int(bool(meta["castling_rights"] & chess.BB_A8)),  # Black queenside
        ]
        additional_features = torch.tensor(
            [move_number] + castling_rights, dtype=torch.float32
        )

        label = torch.tensor(self.scores[idx], dtype=torch.float32)

        return board, additional_features, label
