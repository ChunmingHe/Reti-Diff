import os
import shutil


def rename_and_copy_files(source_folder, destination_train_folder, destination_test_folder):
    # 检查目标文件夹是否存在，如不存在则创建
    if not os.path.exists(destination_train_folder):
        os.makedirs(destination_train_folder)
    if not os.path.exists(destination_test_folder):
        os.makedirs(destination_test_folder)

    # 遍历源文件夹下的所有文件
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)

        filename_split = filename.split('_')
        id = int(filename_split[0])
        if id <= 50:
            new_filename = str(id) + ".png"
            destination_file = os.path.join(destination_train_folder, new_filename)
            # 复制文件到目标文件夹
            shutil.copy(source_file, destination_file)
            print(f"复制文件：{source_file} -> {destination_file}")
        else:
            new_filename = str(id) + ".png"
            destination_file = os.path.join(destination_test_folder, new_filename)
            # 复制文件到目标文件夹
            shutil.copy(source_file, destination_file)
            print(f"复制文件：{source_file} -> {destination_file}")


# 源文件夹路径
source_folder = "/data/fcy/Datasets/Dehaze_Storage/Dense_Haze_NTIRE19/hazy"
# 目标文件夹路径
destination_train_folder = "/data/fcy/Datasets/Dehaze/Dense_Haze/train/hazy"
destination_test_folder = "/data/fcy/Datasets/Dehaze/Dense_Haze/test/hazy"

# 调用函数进行文件重命名和复制
rename_and_copy_files(source_folder, destination_train_folder, destination_test_folder)