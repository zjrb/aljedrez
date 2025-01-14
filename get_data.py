import numpy as np
import os


def load_from_npz(input_path):
    # Load data from an NPZ file
    data = np.load(input_path, allow_pickle=True)
    positions = data["positions"]
    metadata = data["metadata"]
    scores = data["scores"]
    return positions, metadata, scores


def load_multiple_npz(data_dir):
    positions = []
    metadata = []
    scores = []

    # Iterate through all .npz files in the directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".npz"):
            file_path = os.path.join(data_dir, file_name)
            data = np.load(file_path, allow_pickle=True)
            positions.extend(data["positions"])
            metadata.extend(data["metadata"])
            scores.extend(data["scores"])

    return positions, metadata, scores
