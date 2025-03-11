import numpy as np
import time
from queue import PriorityQueue
from utils import Node, get_neighbors, heuristic, is_path_free

def prm(map_data, start, goal, n_points=1000, k_neighbors=10):
    """概率路标图算法实现"""
    start_time = time.time()
    nodes_explored = 0
    visited_nodes = []  # 记录访问过的节点用于可视化
    
    # 生成随机采样点
    points = [start, goal]
    while len(points) < n_points + 2:
        x = np.random.randint(0, map_data.shape[0])
        y = np.random.randint(0, map_data.shape[1])
        if map_data[x, y] > 0.5:  # 确保点在可行区域内
            points.append((x, y))
            visited_nodes.append((x, y))
            nodes_explored += 1
    
    # 构建路标图
    graph = {p: [] for p in points}
    for i, p1 in enumerate(points):
        distances = [(heuristic(p1, p2), p2) for p2 in points[i+1:]]
        distances.sort()
        for _, p2 in distances[:k_neighbors]:
            if is_path_free(p1, p2, map_data):
                graph[p1].append(p2)
                graph[p2].append(p1)
    
    # 在构建好的无向图上使用A*算法在路标图上搜索路径
    path = astar_on_prm(graph, start, goal)
    if path:
        return path, time.time() - start_time, nodes_explored, visited_nodes
    return None, time.time() - start_time, nodes_explored, visited_nodes

def astar_on_prm(graph, start, goal):
    """在PRM图上使用A*算法搜索路径"""
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