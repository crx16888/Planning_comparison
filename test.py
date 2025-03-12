import cv2
import numpy as np

# 加载原始图像
image = cv2.imread(r'C:\Users\95718\Desktop\vscode\Planning\Planning_comparison\maps.png')
if image is None:
    print("错误：无法加载图像")
    exit()

# 检查图像类型
print(f"图像形状：{image.shape}")
if len(image.shape) == 2:
    print("这是灰度图像")
    # 检查是否为黑白图
    unique_values = np.unique(image)
    if len(unique_values) == 2 and (0 in unique_values) and (255 in unique_values):
        print("更具体地说，这是黑白二值图")
    else:
        print("更具体地说，这是真正的灰度图（包含多个灰度级别）")
elif len(image.shape) == 3:
    if image.shape[2] == 3:
        print("这是RGB图像")
    elif image.shape[2] == 4:
        print("这是RGBA图像")
    print(f"图像通道数：{image.shape[2]}")

# 输出详细的像素信息
print("\n像素值统计信息：")
print(f"最小值：{image.min()}")
print(f"最大值：{image.max()}")
print(f"平均值：{image.mean():.2f}")

if len(image.shape) == 3:
    # RGB图像的每个通道的统计
    for i, channel in enumerate(['蓝', '绿', '红']):
        print(f"\n{channel}通道统计：")
        print(f"最小值：{image[:,:,i].min()}")
        print(f"最大值：{image[:,:,i].max()}")
        print(f"平均值：{image[:,:,i].mean():.2f}")