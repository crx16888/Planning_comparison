import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time
from queue import PriorityQueue

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
            if map_data[new_x, new_y] > 0:
                neighbors.append((new_x, new_y))
    return neighbors

def heuristic(pos, goal):
    """启发式函数：使用欧几里得距离"""
    return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

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

def is_path_free(p1, p2, map_data):
    """检查两点之间的路径是否无障碍"""
    x1, y1 = p1
    x2, y2 = p2
    points = np.linspace([x1, y1], [x2, y2], num=20).astype(int)
    return all(map_data[x, y] > 0 for x, y in points)

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

def visualize_results(map_data, start, goal, astar_result, dijkstra_result, astar_prm_result, prm_result):
    """可视化路径规划结果和性能指标"""
    # 计算各算法的路径指标
    metrics = {}
    for name, result in [('A*', astar_result), ('Dijkstra', dijkstra_result), 
                        ('A*-PRM', astar_prm_result), ('PRM', prm_result)]:
        if result and result[0]:
            path_length, smoothness, _ = calculate_path_metrics(result[0])
            metrics[name] = {
                'length': path_length,
                'time': result[1],
                'nodes': result[2],
                'smoothness': smoothness
            }
        else:
            metrics[name] = {
                'length': 0,
                'time': 0,
                'nodes': 0,
                'smoothness': 0
            }
    
    # 路径对比图
    plt.figure(1, figsize=(16, 10))
    plt.imshow(map_data, cmap='binary')
    
    colors = {'A*': ('r', 'pink'), 'Dijkstra': ('b', 'lightblue'), 
              'A*-PRM': ('g', 'lightgreen'), 'PRM': ('y', 'yellow')}
    
    for name, result in [('A*', astar_result), ('Dijkstra', dijkstra_result), 
                        ('A*-PRM', astar_prm_result), ('PRM', prm_result)]:
        if result[0]:
            path = np.array(result[0])
            plt.plot(path[:, 1], path[:, 0], f'{colors[name][0]}-', 
                     linewidth=2, label=f'{name} Path')
            nodes = np.array(result[3])
            plt.scatter(nodes[:, 1], nodes[:, 0], c=colors[name][1], 
                        s=20, alpha=0.3, label=f'{name} Nodes')
    
    plt.plot(start[1], start[0], 'go', markersize=12, label='Start')
    plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')
    plt.title('Path Comparison', fontsize=14, pad=15)
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 调整算法顺序
    algorithm_order = ['Dijkstra', 'A*', 'PRM', 'A*-PRM']
    colors = ['#4B6BFF', '#FF4B4B', '#FFD700', '#4BFF4B']
    
    # 路径长度对比
    plt.figure(2, figsize=(8, 6))
    lengths = [metrics[name]['length'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, lengths, color=colors)
    plt.title('Path Length (Euclidean)', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    plt.xticks(rotation=45)
    
    # 执行时间对比
    plt.figure(3, figsize=(8, 6))
    times = [metrics[name]['time'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, times, color=colors)
    plt.title('Execution Time (seconds)', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    plt.xticks(rotation=45)
    
    # 探索节点数对比
    plt.figure(4, figsize=(8, 6))
    nodes = [metrics[name]['nodes'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, nodes, color=colors)
    plt.title('Nodes Explored', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    plt.xticks(rotation=45)
    
    # 路径平滑度对比
    plt.figure(5, figsize=(8, 6))
    smoothness = [metrics[name]['smoothness'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, smoothness, color=colors)
    plt.title('Path Smoothness (rad)', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    plt.xticks(rotation=45)
    
    plt.show()

def main():
    map_path = r"C:\Users\95718\Desktop\vscode\Planning\Experiment_Algorithm_comparison\maps.png" #这里需要修改成自己的地址
    map_data = load_map(map_path)
    
    # 设置起点和终点
    start = (50, 50)
    goal = (map_data.shape[0]-50, map_data.shape[1]-50)
    
    # 运行算法
    astar_result = astar(map_data, start, goal)
    dijkstra_result = dijkstra(map_data, start, goal)
    astar_prm_result = astar_prm(map_data, start, goal)
    prm_result = prm(map_data, start, goal)
    
    # 可视化结果
    visualize_results(map_data, start, goal, astar_result, dijkstra_result, astar_prm_result, prm_result)

if __name__ == '__main__':
    main()