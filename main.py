import numpy as np
import matplotlib.pyplot as plt

from dijkstra import dijkstra
from astar import astar
from prm import prm
from astar_prm import astar_prm
from utils import calculate_path_metrics
from visualization import visualize_results

def load_map(map_path):
    """加载并处理地图"""
    # 读取图像
    map_data = plt.imread(map_path)
    # 如果是RGB图像，转换为灰度图
    if len(map_data.shape) > 2:
        map_data = np.mean(map_data, axis=2)
    # 二值化处理：将障碍物设为0（黑色），可通行区域设为1（白色）
    map_data = (map_data > 0.5).astype(np.float64)
    return map_data

def main():
    map_path = r"C:\Users\95718\Desktop\vscode\Planning\Planning_comparison\maps.png" #地图文件路径
    map_data = load_map(map_path)
    
    # 设置起点和终点（根据地图大小设置合适的起终点位置）
    start = (50, 50)
    goal = (map_data.shape[0]-50, map_data.shape[1]-50)
    
    # 运行各路径规划算法
    astar_result = astar(map_data, start, goal)
    dijkstra_result = dijkstra(map_data, start, goal)
    astar_prm_result = astar_prm(map_data, start, goal)
    prm_result = prm(map_data, start, goal)
    
    # 可视化结果对比
    visualize_results(map_data, start, goal, astar_result, dijkstra_result, astar_prm_result, prm_result)

if __name__ == '__main__':
    main()