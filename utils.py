import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, pos, g_cost=0, h_cost=0):
        self.pos = pos  # 节点位置 (x, y)
        self.g_cost = g_cost  # 从起点到当前节点的实际代价
        self.h_cost = h_cost  # 从当前节点到目标的估计代价
        self.f_cost = g_cost + h_cost  # 总代价
        self.parent = None  # 父节点，用于路径回溯
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

def get_neighbors(pos, map_data):
    """获取当前位置的有效邻居节点"""
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),  # 4邻域
                 (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 8邻域
    
    for dx, dy in directions:
        new_x, new_y = pos[0] + dx, pos[1] + dy
        # 检查边界
        if 0 <= new_x < map_data.shape[0] and 0 <= new_y < map_data.shape[1]:
            # 检查是否是可通行区域（值为1）
            if map_data[new_x, new_y] > 0.5:
                neighbors.append((new_x, new_y))
    return neighbors

def heuristic(pos, goal):
    """启发式函数：使用欧几里得距离"""
    return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

def is_path_free(p1, p2, map_data):
    """检查两点之间的路径是否无障碍"""
    x1, y1 = p1
    x2, y2 = p2
    points = np.linspace([x1, y1], [x2, y2], num=20).astype(int) #在p1、p2之间均匀生成20个点，并检查这些点是否在障碍物上
    return all(map_data[x, y] > 0 for x, y in points) # 检查20个点是否在障碍上

def path_smoothing(path, map_data):
    """对路径进行平滑处理"""
    if len(path) <= 2:
        return path
    
    smoothed = [path[0]]
    current_point = 0
    
    while current_point < len(path) - 1:
        # 尝试连接当前点和尽可能远的点
        for i in range(len(path) - 1, current_point, -1):
            if is_path_free(path[current_point], path[i], map_data):
                smoothed.append(path[i])
                current_point = i
                break
        else:
            current_point += 1
            if current_point < len(path):
                smoothed.append(path[current_point])
    
    return smoothed

def calculate_path_metrics(path):
    """计算路径的各项指标"""
    if path is None:
        return None, None, None
    
    # 计算总路径长度（欧几里得距离）
    path = np.array(path)
    path_length = 0
    for i in range(len(path)-1):
        path_length += np.sqrt(np.sum((path[i+1] - path[i])**2))
    
    # 计算路径平滑度（相邻路径段的角度变化）
    smoothness = 0
    if len(path) > 2:
        for i in range(len(path)-2):
            v1 = path[i+1] - path[i]
            v2 = path[i+2] - path[i+1]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            smoothness += angle
        smoothness /= (len(path)-2)  # 平均角度变化
    
    # 计算路径安全裕度（到障碍物的最小距离）
    safety_margin = float('inf')  # 这个指标需要地图信息才能计算
    
    return path_length, smoothness, safety_margin