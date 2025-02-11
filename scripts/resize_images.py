from PIL import Image
import os

def crop_and_resize_images(input_folder, output_folder, border_size):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像文件
            image = Image.open(input_path)

            # 获取图像的宽度和高度
            width, height = image.size

            # 计算新的图像尺寸
            new_width = 640
            new_height = 480

            # 裁剪图像
            cropped_image = image.crop((border_size, border_size, width - border_size, height - border_size))

            # 调整图像大小
            resized_image = cropped_image.resize((new_width, new_height))

            # 保存图像到输出文件夹
            resized_image.save(output_path)

            print(f"已保存文件: {output_path}")

def resize_images(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像文件
            image = Image.open(input_path)

            # 计算新的图像尺寸
            new_width = 640
            new_height = 480

            # 调整图像大小
            resized_image = image.resize((new_width, new_height))

            # 保存图像到输出文件夹
            resized_image.save(output_path)

            print(f"已保存文件: {output_path}")



# 定义输入文件夹和输出文件夹的路径
input_folder = "/data0/hcm/DiffSR/Reti-Diff-demotionblur/Datasets/SOTS-Storage/hazy"
output_folder = "/data0/hcm/DiffSR/Reti-Diff-demotionblur/Datasets/SOTS/hazy"

# 调整边框大小
# border_size = 10

# 调用函数来处理图像
resize_images(input_folder, output_folder)