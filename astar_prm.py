import numpy as np
import time
from queue import PriorityQueue
from utils import Node, get_neighbors, heuristic, is_path_free, path_smoothing

def astar_prm(map_data, start, goal, n_points=1000, k_neighbors=10):
    """A*-混合算法实现"""
    start_time = time.time()
    nodes_explored = 0
    visited_nodes = []  # 记录访问过的节点用于可视化
    
    # 启发式采样
    points = [start, goal]
    while len(points) < n_points + 2:
        if np.random.random() < 0.7:  # 70%概率使用启发式采样
            # 在起点和终点之间的区域内进行采样
            x_min = min(start[0], goal[0])
            x_max = max(start[0], goal[0])
            y_min = min(start[1], goal[1])
            y_max = max(start[1], goal[1])
            
            # 在边界附近增加采样概率
            margin = 20
            if np.random.random() < 0.3:
                x = np.random.randint(max(0, x_min - margin), min(map_data.shape[0], x_max + margin))
                y = np.random.randint(max(0, y_min - margin), min(map_data.shape[1], y_max + margin))
            else:
                x = np.random.randint(x_min, x_max + 1)
                y = np.random.randint(y_min, y_max + 1)
        else:  # 30%概率使用随机采样
            x = np.random.randint(0, map_data.shape[0])
            y = np.random.randint(0, map_data.shape[1])
        
        if map_data[x, y] > 0.5:  # 确保点在可行区域内
            points.append((x, y))
            visited_nodes.append((x, y))
            nodes_explored += 1
    
    # 动态路图构建
    graph = {p: [] for p in points}
    for i, p1 in enumerate(points):
        # 计算到目标的启发式值
        h_value = heuristic(p1, goal)
        
        # 根据启发式值动态调整连接数量
        local_k = max(int(k_neighbors * (1 - h_value / heuristic(start, goal))), 3)
        
        # 对候选邻居进行排序
        distances = []
        for p2 in points[i+1:]:
            dist = heuristic(p1, p2)
            h_diff = abs(heuristic(p2, goal) - h_value)  # 启发式值的差异
            # 综合考虑距离和启发式值
            score = dist + 0.5 * h_diff
            distances.append((score, p2))
        
        distances.sort()
        for _, p2 in distances[:local_k]:
            if is_path_free(p1, p2, map_data):
                graph[p1].append(p2)
                graph[p2].append(p1)
    
    # 使用A*在路图上搜索路径
    path = astar_on_graph(graph, start, goal)
    if path:
        # 路径平滑处理
        smoothed_path = path_smoothing(path, map_data)
        return smoothed_path, time.time() - start_time, nodes_explored, visited_nodes
    
    return None, time.time() - start_time, nodes_explored, visited_nodes

def astar_on_graph(graph, start, goal):
    """在构建的路图上使用A*算法搜索路径"""
    open_set = PriorityQueue()
    start_node = Node(start, 0, heuristic(start, goal))
    open_set.put((0, start_node))
    
    closed_set = {start: start_node}
    
    while not open_set.empty():
        _, current = open_set.get()
        
        if current.pos == goal:
            path = []
            while current.pos != start:
                path.append(current.pos)
                current = current.parent
            path.append(start)
            path.reverse()
            return path
        
        for neighbor_pos in graph[current.pos]:
            new_g_cost = current.g_cost + heuristic(current.pos, neighbor_pos)
            
            if neighbor_pos not in closed_set:
                neighbor = Node(neighbor_pos, new_g_cost, heuristic(neighbor_pos, goal))
                neighbor.parent = current
                open_set.put((neighbor.f_cost, neighbor))
                closed_set[neighbor_pos] = neighbor
            elif new_g_cost < closed_set[neighbor_pos].g_cost:
                neighbor = closed_set[neighbor_pos]
                neighbor.g_cost = new_g_cost
                neighbor.f_cost = new_g_cost + neighbor.h_cost
                neighbor.parent = current
    
    return None