import numpy as np
import time
from queue import PriorityQueue
from utils import Node, get_neighbors, heuristic

def astar(map_data, start, goal):
    """A*算法实现"""
    start_time = time.time()
    nodes_explored = 0
    visited_nodes = []  # 记录访问过的节点用于可视化
    
    open_set = PriorityQueue() #已经被发现但是尚未处理的节点
    start_node = Node(start)
    open_set.put((0, start_node))
    
    closed_set = {start: start_node} #已经处理过的节点及其最优路径，当发现更优路径时再次加入开放集
    
    while not open_set.empty():
        _, current = open_set.get()
        nodes_explored += 1
        visited_nodes.append(current.pos)
        
        if current.pos == goal:
            path = []
            while current.pos != start:
                path.append(current.pos)
                current = current.parent
            path.append(start)
            path.reverse()
            return path, time.time() - start_time, nodes_explored, visited_nodes
        
        for neighbor_pos in get_neighbors(current.pos, map_data):
            new_g_cost = current.g_cost + 1
            
            if neighbor_pos not in closed_set: #如果节点没有被发现，那么加入开放集处理
                neighbor = Node(neighbor_pos, new_g_cost, heuristic(neighbor_pos, goal))
                neighbor.parent = current
                open_set.put((neighbor.f_cost, neighbor))
                closed_set[neighbor_pos] = neighbor
            elif new_g_cost < closed_set[neighbor_pos].g_cost: #如果节点已经被发现过了，那么看这次的是不是最优路径，是的话再次加入开放集
                neighbor = closed_set[neighbor_pos]
                neighbor.g_cost = new_g_cost
                neighbor.f_cost = new_g_cost + neighbor.h_cost
                neighbor.parent = current
                open_set.put((neighbor.f_cost, neighbor))
    
    return None, time.time() - start_time, nodes_explored, visited_nodes