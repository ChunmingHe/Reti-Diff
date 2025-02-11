import os
import shutil

# 源文件夹路径
source_folder = '/data0/hcm/DiffSR/Reti-Diff-demotionblur/Datasets/ITS_v2/hazy'
# 目标文件夹路径
destination_folder = '/data0/hcm/DiffSR/Reti-Diff-demotionblur/Datasets/ITS_v2/hazy_renamed'

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    source_file = os.path.join(source_folder, filename)

    # 获取文件名和后缀名
    file_name, file_ext = os.path.splitext(filename)

    # 按照_分割文件名
    file_parts = file_name.split('_')

    # 判断文件名是否符合格式要求
    if len(file_parts) >= 3:
        # 获取第二个_之前的内容
        new_file_name = '_'.join(file_parts[:2]) + file_ext

        # 构建目标文件路径
        destination_file = os.path.join(destination_folder, new_file_name)

        print(destination_file)

        # 复制文件到目标文件夹
        shutil.copyfile(source_file, destination_file)