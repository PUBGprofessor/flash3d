import pickle
import gzip
from pathlib import Path


def load_seq_data(data_path, split):
    file_path = data_path / f"{split}.pickle.gz"
    with gzip.open(file_path, "rb") as f:
        seq_data = pickle.load(f)
    return seq_data

path = Path(r"F:\3DGS_code\dataset\RealEstate10K")
seq  = load_seq_data(path, "test")
# print(seq['ffd8aa5c07187add']['poses'])
print(seq.keys())