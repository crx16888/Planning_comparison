import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_path_metrics

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