import numpy as np
import matplotlib.pyplot as plt

from dijkstra import dijkstra
from astar import astar
from prm import prm
from astar_prm import astar_prm
from dqn import dqn  # 导入DQN算法
from utils import calculate_path_metrics
from visualization import visualize_results

# 处理RGB图而非灰度图和黑白图，详见test.py
def load_map(map_path):
    """加载并处理地图"""
    # 读取图像
    map_data = plt.imread(map_path)
    
    # 如果是RGB图像，转换为灰度图
    if len(map_data.shape) > 2:
        original_data = map_data.copy()  # 保存原始数据
        map_data = np.mean(map_data, axis=2) #对三通道的像素值做平均
        
        # 检查转换前后的值是否相等
        is_equal = np.all(original_data[:,:,0] == original_data[:,:,1]) and \
                  np.all(original_data[:,:,1] == original_data[:,:,2])
        print("\nRGB三个通道是否完全相等:", is_equal)
        if is_equal:
            print("这是一张伪彩色图像，每个通道的值都相同")
        else:
            print("这是一张真彩色图像，不同通道的值不同")
    # print(map_data[0])
    # 处理灰度图：只将像素值为0的位置标记为障碍物（0），其他位置为可通行区域（1）
    map_data = (map_data > 0.998).astype(np.int_)
    # print(map_data[0])
    # 验证处理结果
    print("\n处理后的数据：")
    print("- 唯一值:", np.unique(map_data))  # 应该只显示[0, 1]
    #通过这一系列操作强行把图像转换为了一个只含[0,1]的灰度图像
    return map_data

def main():
    map_path = r"C:\Users\95718\Desktop\vscode\Planning\Planning_comparison\maps.png" #地图文件路径
    map_data = load_map(map_path)
    print(map_data.shape)  # 打印地图尺寸
    # 设置起点和终点（根据地图大小设置合适的起终点位置）
    start = (50, 50)
    goal = (map_data.shape[0]-50, map_data.shape[1]-50)
    
    # 运行各路径规划算法
    astar_result = astar(map_data, start, goal)
    dijkstra_result = dijkstra(map_data, start, goal)
    astar_prm_result = astar_prm(map_data, start, goal)
    prm_result = prm(map_data, start, goal)
    dqn_result = dqn(map_data, start, goal)  # 运行DQN算法
    
    # 可视化结果对比
    visualize_results(map_data, start, goal, astar_result, dijkstra_result, astar_prm_result, prm_result, dqn_result)

if __name__ == '__main__':
    main()