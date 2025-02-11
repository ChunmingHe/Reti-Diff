import os
import shutil

# 源文件夹路径
source_folder = '/data/fcy/Datasets/Dehaze/SOTS/indoor/clear'
# 目标文件夹路径
destination_folder = '/data/fcy/Datasets/Dehaze/SOTS/indoor/clear_copied'

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    source_file = os.path.join(source_folder, filename)

    # 获取文件名和后缀名
    file_name, file_ext = os.path.splitext(filename)

    # 复制文件到目标文件夹，并按照要求重命名
    for i in range(1, 11):
        new_file_name = f"{file_name}_{i}{file_ext}"
        destination_file = os.path.join(destination_folder, new_file_name)
        shutil.copyfile(source_file, destination_file)