import os
import chess.pgn
import chess.engine
import numpy as np
import h5py

piece_to_index = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
    ".": 12,
}

def encode_board(board):
    encoded = np.zeros((8, 8, 13), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        rank, file = divmod(square, 8)
        if piece:
            encoded[rank, file, piece_to_index[piece.symbol()]] = 1
        else:
            encoded[rank, file, 12] = 1
    return encoded


def save_to_hdf5(output_path, positions, metadata, scores, chunk_size=1000):
    with h5py.File(output_path, "w") as f:
        # Create datasets for positions, metadata, and scores
        f.create_dataset(
            "positions",
            data=positions,
            chunks=True,
            compression="gzip",
            compression_opts=9,
        )
        f.create_dataset(
            "metadata",
            data=metadata,
            chunks=True,
            compression="gzip",
            compression_opts=9,
        )
        f.create_dataset(
            "scores", data=scores, chunks=True, compression="gzip", compression_opts=9
        )
    print(f"Saved data to {output_path}")



def save_partial_data(positions, metadata, scores, output_dir, idx):
    output_file = os.path.join(output_dir, f"chunk_{idx}.h5")
    save_to_hdf5(output_file, positions, metadata, scores)
    positions.clear()
    metadata.clear()
    scores.clear()


def process_single_pgn(pgn_file, stockfish_path,output_dir, idx):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    positions = []
    metadata = []
    scores = []
    with open(pgn_file) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if not game:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                info = engine.analyse(board, chess.engine.Limit(depth=18))
                scores.append(info["score"].white().score(mate_score=10000))
                positions.append(encode_board(board))
                metadata.append(
                    {
                        "move_number": board.fullmove_number,
                        "castling_rights": board.castling_rights,
                    }
                )
    save_partial_data(positions, metadata, scores, output_directory, )
    engine.quit()


def process_pgns_with_progress(
    pgn_dir, stockfish_path, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    pgn_files = [
        os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.endswith(".pgn")
    ]
    for idx,pgn_file in enumerate(pgn_files):
        process_single_pgn(pgn_file, stockfish_path, output_dir, idx)


if __name__ == "__main__":
    STOCKFISH_PATH = (
        "/Users/zacharybeach/Downloads/stockfish/stockfish-macos-m1-apple-silicon"
    )
    pgn_directory = "cclr/train"
    output_directory = "processed_data/train"
    process_pgns_with_progress(
        pgn_directory, STOCKFISH_PATH, output_directory
    )
