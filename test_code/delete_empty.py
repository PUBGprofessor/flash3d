import os

def delete_empty_dirs(directory):
    # 遍历目录中的所有子目录
    count = 0
    # 获取目录下的所有子目录
    for dirname in os.listdir(directory):
        dir_to_check = os.path.join(directory, dirname)
        if os.path.isdir(dir_to_check) and not os.listdir(dir_to_check):  # 如果是目录且为空
            os.rmdir(dir_to_check)  # 删除空目录
            count += 1
            print(f"Deleted empty directory: {dir_to_check}")
    print(f"Total empty directories deleted: {count}")

# 设置你想清理的目录
directory_to_clean = './data/RealEstate10K/test'

delete_empty_dirs(directory_to_clean)
