from pathlib import Path
import gzip
import pickle
import argparse
from tqdm import tqdm

from datasets.re10k import load_seq_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", type=str)
    parser.add_argument("-d", "--data_path", type=str)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    split = args.split
    seq_data = load_seq_data(data_path, split) # 读取数据集的索引文件，返回一个字典，包含每个序列的信息
    seq_keys = list(seq_data.keys())
    for seq in tqdm(seq_keys):  # tqdm(seq_keys) 用于在迭代时显示进度条，帮助跟踪循环的执行进度。
        if not data_path.joinpath(split, seq).is_dir():
            print(f"missing sequence {seq}")
            del seq_data[seq]
        else:
            path = data_path.joinpath(split, seq)
            for i in range(len(seq_data[seq]['timestamps'])):
                if not path.joinpath(f"{seq_data[seq]['timestamps'][i]}.jpg").exists():
                    print(f"missing image {path.joinpath(f'{i:06d}.jpg')}")
                    del seq_data[seq]
                    break
    # del seq_data["829c847536f4ed7c"]

    file_path = data_path / f"{split}.pickle.gz" # 索引文件
    with gzip.open(file_path, "wb") as f:
        pickle.dump(seq_data, f)


if __name__ == "__main__":
    main()
