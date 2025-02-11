from PIL import Image
import os

def pad_images(input_folder, output_folder, border_size):
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

            # 设置pad的大小和颜色
            pad_size = 10  # pad的大小，单位为像素
            pad_color = (255, 255, 255)  # pad的颜色，这里使用白色（RGB值为255, 255, 255）

            # 计算新图像的尺寸
            new_width = image.width + 2 * pad_size
            new_height = image.height + 2 * pad_size

            # 创建一个新的图像对象，尺寸为新的尺寸，并用指定的颜色填充
            padded_image = Image.new(image.mode, (new_width, new_height), pad_color)

            # 将原始图像粘贴到新图像对象中心位置
            padded_image.paste(image, (pad_size, pad_size))

            # 保存图像到输出文件夹
            padded_image.save(output_path)

            print(f"已保存文件: {output_path}")




# 定义输入文件夹和输出文件夹的路径
input_folder = "/data/fcy/Datasets/Dehaze/SOTS/indoor/backup/hazy"
output_folder = "//data/fcy/Datasets/Dehaze/SOTS/indoor/hazy"

# 调整边框大小
border_size = 10

# 调用函数来处理图像
pad_images(input_folder, output_folder, border_size)