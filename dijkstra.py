import numpy as np
import time
from queue import PriorityQueue
from utils import Node, get_neighbors

def dijkstra(map_data, start, goal):
    """Dijkstra算法实现"""
    start_time = time.time()
    nodes_explored = 0
    visited_nodes = [] #记录访问过的节点用于可视化
    
    open_set = PriorityQueue() #优先队列：已经被发现但是尚未处理的节点，按g_cost排列
    start_node = Node(start) 
    open_set.put((0, start_node)) #将起点加入优先队列，代价为0
    
    closed_set = {start: start_node} #已经处理过的节点及其最优路径，当发现更优路径时再次加入开放集
    
    while not open_set.empty():
        _, current = open_set.get() #获取代价最小的节点（节点是一个对象）
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
            new_g_cost = current.g_cost + 1 #计算新的代价
            
            if neighbor_pos not in closed_set: #如果是没有访问过的节点
                neighbor = Node(neighbor_pos, new_g_cost)
                neighbor.parent = current
                open_set.put((new_g_cost, neighbor)) #加入优先队列
                closed_set[neighbor_pos] = neighbor
            elif new_g_cost < closed_set[neighbor_pos].g_cost: #如果是已经访问过的节点，判断是否需要更新（这是否是更优的路径）
                neighbor = closed_set[neighbor_pos]
                neighbor.g_cost = new_g_cost
                neighbor.parent = current
                open_set.put((new_g_cost, neighbor)) #更新，加入优先队列（优先队列处理后会加入封闭集）
    
    return None, time.time() - start_time, nodes_explored, visited_nodes