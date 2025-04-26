"""
Author: Felix Wimbauer
Source: https://github.com/Brummi/BehindTheScenes/blob/main/datasets/realestate10k/download_realestate10k.py
"""
import argparse
import os
import sys
import subprocess
from multiprocessing import Pool
from pathlib import Path
from time import sleep

from pytubefix import YouTube
import tqdm
from subprocess import call

import pickle
import yt_dlp

def download_video(url, cookies_path, output_path):
    # yt-dlp 命令
    ydl_cmd = [
        'yt-dlp', 
        '--cookies', cookies_path,  # cookies 文件路径
        '--format', 'bestvideo[height=360]',  # 下载 360p 视频
        '--output', output_path,  # 设置输出路径
        '--no-check-certificate',  # 不验证证书
        url  # YouTube 视频 URL
    ]
    
    try:
        # 调用 yt-dlp 命令
        result = subprocess.run(ydl_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}", file=sys.stderr)

class Data:
    def __init__(self, url, seqname, list_timestamps):
        self.url = url
        self.list_seqnames = []
        self.list_list_timestamps = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


def process(data, seq_id, videoname, output_root):
    seqname = data.list_seqnames[seq_id]
    out_path = output_root / seqname
    if not out_path.exists():
        out_path.mkdir(exist_ok=True, parents=True)
    else:
        print("[INFO] {} already exists, skip process".format(seqname))
        # print("[INFO] Something Wrong, stop process")
        return True

    list_str_timestamps = []
    for timestamp in data.list_list_timestamps[seq_id]:
        timestamp = int(timestamp / 1000)
        str_hour = str(int(timestamp / 3600000)).zfill(2)
        str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
        str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
        str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
        _str_timestamp = str_hour + ":" + str_min + ":" + str_sec + "." + str_mill
        list_str_timestamps.append(_str_timestamp)

    # extract frames from a video
    for idx, str_timestamp in enumerate(list_str_timestamps):
        call(("ffmpeg", "-ss", str_timestamp, "-i", str(videoname), "-vframes", "1", "-f", "image2", str(out_path / f'{data.list_list_timestamps[seq_id][idx]}.jpg')), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return False


def wrap_process(list_args):
    return process(*list_args)


class DataDownloader:
    def __init__(self, data_path: Path, out_path: Path, tmp_path: Path, mode='test'):
        print("[INFO] Loading data list ... ", end='')
        self.data_path = data_path
        self.out_path = out_path # data/RealEstate10K/train or data/RealEstate10K/test
        self.tmp_path = tmp_path # tmpdir
        self.mode = mode

        # self.is_done = out_path.exists() # true
        self.is_done = False
        
        dataloader_path = data_path.parent / ("DataDownloader_" + mode)
        # 如果已经处理过数据，则直接加载
        if dataloader_path.exists():
            with open(dataloader_path, 'rb') as f:
                self.list_data = pickle.load(f)
            print(" Done! ")
            print("[INFO] {} movies are used in {} mode".format(len(self.list_data), self.mode))
            return

        self.list_seqnames = sorted(self.data_path.glob('*.txt')) # 列表：data/RealEstate10K/train/*.txt or data/RealEstate10K/test/*.txt


        out_path.mkdir(exist_ok=True, parents=True)

        self.list_data = {} # 字典：{youtube_url: Data()} 每个youtube_url可能有多个场景
        for txt_file in tqdm.tqdm(self.list_seqnames):
            seq_name = txt_file.stem # 去掉后缀名，得到txt的文件名

            # extract info from txt
            with open(txt_file, "r") as seq_file:
                lines = seq_file.readlines()
                youtube_url = ""
                list_timestamps = []
                for idx, line in enumerate(lines):
                    if idx == 0:
                        youtube_url = line.strip()
                    else:
                        timestamp = int(line.split(' ')[0])
                        list_timestamps.append(timestamp)

            if youtube_url in self.list_data:
                self.list_data[youtube_url].add(seq_name, list_timestamps)
            else:
                self.list_data[youtube_url] = Data(youtube_url, seq_name, list_timestamps)

        # 保存数据到本地
        with open(dataloader_path, 'wb') as f:
            pickle.dump(self.list_data, f)

        print(" Done! ")
        print("[INFO] {} movies are used in {} mode".format(len(self.list_data), self.mode))

    def run(self):
        print("[INFO] Start downloading {} movies".format(len(self.list_data)))

        failed_videos_path = os.path.join(str(self.data_path.parent), 'failed_videos_' + self.mode + '.txt')
        # 创建一个空集合来存储所有的字符串
        failed_videos_set = set()

        # 读取文件并将每行数据添加到集合中
        with open(failed_videos_path, 'r') as file:
            for line in file:
                # 去掉每行末尾的换行符，添加到集合中
                failed_videos_set.add(line.strip())


        sum_url = len(self.list_data)
        url_count = 0 # 处理过的url数量，不论成功与否
        restart_count = 0 # 处理过的url数量，成功的url数量

        # 读取上次中断的url数量
        save_restart_path = "./data/RealEstate10K/restart.txt"
        if os.path.exists(save_restart_path):
            with open(save_restart_path, 'r') as f:
                restart_count = int(f.read())
                print(f"[INFO] Restart from {restart_count} url")
        
        for global_count, data in enumerate(self.list_data.values()):
            if url_count < restart_count: # 老是中断，从断点开始下
                url_count += 1
                print(f"[INFO] {url_count}/{sum_url} movies are downloaded")
                continue
            
            with open(save_restart_path, 'w') as f:
                f.write(str(url_count))
            
            print("[INFO] Downloading {} ".format(data.url))
            current_file = self.tmp_path / f"current_{self.mode}"

            is_done = True
            for seqname in data.list_seqnames: # 判断url中的所有场景是否已经下载过或下载失败
                if seqname in failed_videos_set: # 如果已经下载失败，则直接跳过
                    is_done = True
                    break
                if not os.path.exists(os.path.join(str(self.out_path), seqname)):  # 如果场景文件夹不存在，则说明没有下载过
                    is_done = False
                    break
            
            if is_done:
                print("[INFO] {} already exists, skip download".format(data.url))
                url_count += 1
                print(f"[INFO] {url_count}/{sum_url} movies are downloaded")
                continue

            call(("rm", "-r", str(current_file)))
            # current_file.mkdir(exist_ok=True, parents=True)
            try:
                # # sometimes this fails because of known issues of pytube and unknown factors
                # yt = YouTube(data.url, use_oauth=True)
                # stream = yt.streams.filter(res='360p').first()
                # stream.download(str(current_file))
                os.mkdir(current_file)
                download_video(data.url, r'./test_code/www.youtube.com_cookies.txt', r"./tmpdir/current_test/1.mp4")

            except Exception as e:
                # Print the error message and traceback
                print("[ERROR] An error occurred: ", e)
                # traceback.print_exc()

                with open(os.path.join(str(self.data_path.parent), 'failed_videos_' + self.mode + '.txt'), 'a') as f:
                    for seqname in data.list_seqnames:
                        f.writelines(seqname + '\n')
                url_count += 1
                continue

            sleep(1)

            try:
                current_file = next(current_file.iterdir())
            except StopIteration:
                pass

            if len(data) == 1:  # len(data) is len(data.list_seqnames)
                process(data, 0, current_file, self.out_path)
            else:
                with Pool(processes=4) as pool:
                    pool.map(wrap_process, [(data, seq_id, current_file, self.out_path) for seq_id in range(len(data))])

            print(f"[INFO] Extracted {sum(map(len, data.list_list_timestamps))}")

            # remove videos
            call(("rm", str(current_file)))
            # os.system(command)

            url_count += 1
            print(f"[INFO] {url_count}/{sum_url} movies are downloaded")
            if self.is_done:
                return False

        return True

    def show(self):
        print("########################################")
        global_count = 0
        for data in self.list_data.values():
            # print(" URL : {}".format(data.url))
            for idx in range(len(data)):
                # print(" SEQ_{} : {}".format(idx, data.list_seqnames[idx]))
                # print(" LEN_{} : {}".format(idx, len(data.list_list_timestamps[idx])))
                global_count = global_count + 1
            # print("----------------------------------------")

        print("TOTAL : {} sequnces".format(global_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str) # train or test
    parser.add_argument("-d", "--data_path", type=str) # data/RealEstate10K
    parser.add_argument("-o", "--out_path", type=str) # data/RealEstate10K
    tmpdir = os.environ.get('TMPDIR')
    parser.add_argument("-t", "--tmp_path", default='tmpdir', type=str)

    args = parser.parse_args()
    mode = args.mode
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    tmp_path = Path(args.tmp_path)


    if mode not in ["test", "train"]:
        raise ValueError(f"Invalid split mode: {mode}")

    data_path = data_path / mode # data/RealEstate10K/train or data/RealEstate10K/test
    out_path = out_path / mode # data/RealEstate10K/train or data/RealEstate10K/test
    downloader = DataDownloader(
        data_path=data_path,
        out_path=out_path,
        tmp_path=tmp_path,
        mode=mode)

    

    downloader.show()
    is_ok = downloader.run()

    if is_ok:
        print("Done!")
    else:
        print("Failed")


if __name__ == "__main__":
    main()


