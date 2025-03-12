import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_path_metrics

def visualize_results(map_data, start, goal, astar_result, dijkstra_result, astar_prm_result, prm_result):
    """可视化路径规划结果和性能指标"""
    # 计算各算法的路径指标
    metrics = {} # 创建空字典，用于存储各算法的性能指标    
    for name, result in [('A*', astar_result), ('Dijkstra', dijkstra_result), 
                        ('A*-PRM', astar_prm_result), ('PRM', prm_result)]:
        # result分别代表路径点列表、算法执行的时间、算法探索过的节点总数、所有探索过的节点的位置坐标列表
        if result and result[0]:
            path_length, smoothness, _ = calculate_path_metrics(result[0]) #获取路径长度和平滑度
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
    # 路径对比图
    plt.figure(1, figsize=(16, 10)) # 创建图形窗口，设置大小
    plt.imshow(map_data, cmap='gray') # 显示灰度图像
    
    # 为每个算法定义两种颜色：(路径颜色, 探索节点颜色)
    colors = {'A*': ('r', 'pink'), 
              'Dijkstra': ('b', 'lightblue'),  
              'A*-PRM': ('g', 'lightgreen'), 
              'PRM': ('y', 'yellow')}  
    
    for name, result in [('A*', astar_result), ('Dijkstra', dijkstra_result), 
                        ('A*-PRM', astar_prm_result), ('PRM', prm_result)]:
        if result[0]: #检查路径点列表判断确实有路径存在
            path = np.array(result[0])
            plt.plot(path[:, 1], path[:, 0], color=colors[name][0], 
                    linewidth=2.5, label=f'{name} Path')  # 绘制路径 path[:, 1]表示第二列的所有行，提取所有路径点的列坐标（x坐标）
            nodes = np.array(result[3])
            plt.scatter(nodes[:, 1], nodes[:, 0], c=colors[name][1], 
                      s=10, alpha=0, label=f'{name} Nodes') #绘制探索节点的散点图（这里考虑要不要）
    
    plt.plot(start[1], start[0], 'go', markersize=12, label='Start')
    plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')
    plt.title('Path Comparison', fontsize=16, pad=15, fontname='Times New Roman')
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), prop={'family': 'Times New Roman'}) #绘制起点、终点、标题和图例
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    
    # 调整算法展示顺序
    algorithm_order = ['Dijkstra', 'A*', 'PRM', 'A*-PRM']
    colors = ['#4B6BFF', '#FF4B4B', '#FFD700', '#4BFF4B']
    
    # 路径长度对比
    plt.figure(2, figsize=(10, 6))
    lengths = [metrics[name]['length'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, lengths, color=colors, width=0.6) #绘制柱状图
    plt.title('Path Length (Euclidean)', fontsize=16, pad=15, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.2, axis='y') #只显示y轴网格线，降低透明度
    plt.tick_params(axis='both', which='major', labelsize=12)
    for bar in bars: #获取每个柱子的高度并在其上方中间位置标注
        height = bar.get_height()
        plt.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=12,
                    fontname='Times New Roman')
    plt.xticks(rotation=45, fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.ylabel('Length', fontsize=14, fontname='Times New Roman')
    
    # 执行时间对比
    plt.figure(3, figsize=(10, 6))
    times = [metrics[name]['time'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, times, color=colors, width=0.6)
    plt.title('Execution Time (seconds)', fontsize=16, pad=15, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=12,
                    fontname='Times New Roman')
    plt.xticks(rotation=45, fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.ylabel('Time (s)', fontsize=14, fontname='Times New Roman')
    
    # 探索节点数对比
    plt.figure(4, figsize=(10, 6))
    nodes = [metrics[name]['nodes'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, nodes, color=colors, width=0.6)
    plt.title('Nodes Explored', fontsize=16, pad=15, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=12,
                    fontname='Times New Roman')
    plt.xticks(rotation=45, fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.ylabel('Number of Nodes', fontsize=14, fontname='Times New Roman')
    
    # 路径平滑度对比
    plt.figure(5, figsize=(10, 6))
    smoothness = [metrics[name]['smoothness'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, smoothness, color=colors, width=0.6)
    plt.title('Path Smoothness (rad)', fontsize=16, pad=15, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=12,
                    fontname='Times New Roman')
    plt.xticks(rotation=45, fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.ylabel('Smoothness (rad)', fontsize=14, fontname='Times New Roman')
    
    plt.show()