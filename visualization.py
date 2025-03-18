import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_path_metrics

class DQNVisualizer:
    def __init__(self):
        self.episode_rewards = [0]  # 初始化为0
        self.epsilon_history = []
        self.losses = []
        self.current_path = []
        self.best_path = None
        self.best_reward = float('-inf')
        self.visited_positions = set()
        
        # 创建实时显示的图表
        plt.ion()
        # 修改为2x2布局以容纳所有图表
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('DQN Training Progress', fontsize=16, fontname='Times New Roman')
        
        # 初始化所有子图
        self.path_ax = self.axes[0, 0]      # 路径搜索过程
        self.epsilon_ax = self.axes[0, 1]    # 探索率曲线
        self.loss_ax = self.axes[1, 0]       # 损失函数曲线
        self.reward_ax = self.axes[1, 1]     # 奖励曲线

    def update(self, episode, reward, epsilon, loss, steps, current_pos=None, map_data=None, start=None, goal=None):
        """更新训练数据并刷新显示"""
        self.episode_rewards.append(reward)
        self.epsilon_history.append(epsilon)
        if loss is not None:
            self.losses.append(loss)
        
        if current_pos is not None:
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_path = self.current_path.copy()
            self.current_path = []
            self.current_path.append(current_pos)
        
        # 设置全局样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 清空并设置背景色
        for ax in self.axes.flat:
            ax.clear()
            ax.set_facecolor('#f8f9fa')
        
        # 设置全局字体
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
        
        # 绘制路径搜索过程
        if map_data is not None:
            self.path_ax.imshow(map_data, cmap='binary')
            # 只绘制最佳路径
            if self.best_path:
                path_x = [pos[0] for pos in self.best_path]
                path_y = [pos[1] for pos in self.best_path]
                self.path_ax.plot(path_y, path_x, 'b-', linewidth=2, alpha=0.8, label='Best Path')
            # 绘制起点和终点
            if start is not None:
                self.path_ax.plot(start[1], start[0], 'go', markersize=10, label='Start')
            if goal is not None:
                self.path_ax.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
            # 绘制当前位置
            if current_pos is not None:
                self.path_ax.plot(current_pos[1], current_pos[0], 'yo', markersize=8)
            
            self.path_ax.set_title('Path Search Process', pad=10)
            self.path_ax.legend()
            self.path_ax.grid(False)
        
        # 绘制探索率曲线
        self.epsilon_ax.plot(self.epsilon_history, color='#66C2A5', linewidth=1.5, 
                            label='Epsilon', alpha=0.8)
        self.epsilon_ax.fill_between(range(len(self.epsilon_history)), 
                                    self.epsilon_history, alpha=0.2, 
                                    color='#66C2A5')
        self.epsilon_ax.set_title('Exploration Rate (ε)', pad=10)
        self.epsilon_ax.set_xlabel('Episodes')
        self.epsilon_ax.set_ylabel('Epsilon Value')
        self.epsilon_ax.grid(True, linestyle='--', alpha=0.7)
        self.epsilon_ax.spines['top'].set_visible(False)
        self.epsilon_ax.spines['right'].set_visible(False)
        
        # 绘制损失函数曲线
        if self.losses:
            self.loss_ax.plot(self.losses, color='#FC8D62', linewidth=1.5, 
                                label='Loss', alpha=0.8)
            self.loss_ax.fill_between(range(len(self.losses)), 
                                        self.losses, alpha=0.2, 
                                        color='#FC8D62')
            self.loss_ax.set_title('Training Loss', pad=10)
            self.loss_ax.set_xlabel('Training Steps')
            self.loss_ax.set_ylabel('Loss Value')
            self.loss_ax.grid(True, linestyle='--', alpha=0.7)
            self.loss_ax.spines['top'].set_visible(False)
            self.loss_ax.spines['right'].set_visible(False)
        
        # 绘制奖励曲线
        self.reward_ax.plot(self.episode_rewards, color='#8DA0CB', linewidth=1.5, 
                            label='Reward', alpha=0.8)
        self.reward_ax.fill_between(range(len(self.episode_rewards)), 
                                    self.episode_rewards, alpha=0.2, 
                                    color='#8DA0CB')
        self.reward_ax.set_title('Episode Rewards', pad=10)
        self.reward_ax.set_xlabel('Episodes')
        self.reward_ax.set_ylabel('Reward Value')
        self.reward_ax.grid(True, linestyle='--', alpha=0.7)
        self.reward_ax.spines['top'].set_visible(False)
        self.reward_ax.spines['right'].set_visible(False)
        
        # 添加当前训练信息
        loss_text = f'{loss:.4f}' if loss is not None else 'N/A'
        info_text = (
            f'Current Episode: {episode}\n'
            f'Reward: {reward:.2f}\n'
            f'Epsilon: {epsilon:.3f}\n'
            f'Loss: {loss_text}\n'
            f'Steps: {steps}'
        )
        
        # 使用专业的文本框样式
        plt.figtext(0.02, 0.02, info_text,
                    bbox=dict(facecolor='white', alpha=0.8, 
                             edgecolor='#CCCCCC', boxstyle='round,pad=1'),
                    fontsize=9, family='Times New Roman')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=3.0, w_pad=3.0)
        
        # 刷新显示
        plt.pause(0.2)
    
    def save(self, filename='training_results.png'):
        """保存训练结果图表"""
        plt.savefig(filename)
    
    def close(self):
        """关闭图表"""
        plt.close(self.fig)
        plt.ioff()

def visualize_results(map_data, start, goal, astar_result, dijkstra_result, astar_prm_result, prm_result, dqn_result):
    """可视化路径规划结果和性能指标"""
    # 计算各算法的路径指标
    metrics = {}
    for name, result in [('A*', astar_result), ('Dijkstra', dijkstra_result), 
                        ('A*-PRM', astar_prm_result), ('PRM', prm_result), ('DQN', dqn_result)]:
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
    plt.imshow(map_data, cmap='gray')
    
    # 为每个算法定义颜色
    colors = {'A*': ('r', 'pink'), 
              'Dijkstra': ('b', 'lightblue'),  
              'A*-PRM': ('g', 'lightgreen'), 
              'PRM': ('y', 'yellow'),
              'DQN': ('m', 'plum')}
    
    for name, result in [('A*', astar_result), ('Dijkstra', dijkstra_result), 
                        ('A*-PRM', astar_prm_result), ('PRM', prm_result), ('DQN', dqn_result)]:
        if result[0]:
            path = np.array(result[0])
            plt.plot(path[:, 1], path[:, 0], color=colors[name][0], 
                    linewidth=2.5, label=f'{name} Path')
            nodes = np.array(result[3])
            plt.scatter(nodes[:, 1], nodes[:, 0], c=colors[name][1], 
                      s=10, alpha=0, label=f'{name} Nodes')
    
    plt.plot(start[1], start[0], 'go', markersize=12, label='Start')
    plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')
    plt.title('Path Comparison', fontsize=16, pad=15, fontname='Times New Roman')
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), prop={'family': 'Times New Roman'})
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    
    # 性能指标对比图
    algorithm_order = ['Dijkstra', 'A*', 'PRM', 'A*-PRM', 'DQN']
    colors = ['#4B6BFF', '#FF4B4B', '#FFD700', '#4BFF4B', '#FF00FF']
    
    # 路径长度对比
    plt.figure(2, figsize=(10, 6))
    lengths = [metrics[name]['length'] for name in algorithm_order]
    bars = plt.bar(algorithm_order, lengths, color=colors, width=0.6)
    plt.title('Path Length (Euclidean)', fontsize=16, pad=15, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.2, axis='y')
    plt.tick_params(axis='both', which='major', labelsize=12)
    for bar in bars:
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
        plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=12,
                    fontname='Times New Roman')
    plt.xticks(rotation=45, fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.ylabel('Nodes', fontsize=14, fontname='Times New Roman')
    
    plt.tight_layout()
    plt.show()