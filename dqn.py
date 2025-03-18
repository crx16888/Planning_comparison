import numpy as np
import time
from utils import DQNAgent
from visualization import DQNVisualizer

def train_dqn(map_data, start, goal, max_episodes=500, max_steps=500):
    """使用DQN算法进行路径规划
    
    参数:
        map_data: 地图数据，0表示障碍物，1表示可通行区域
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)
        max_episodes: 最大训练回合数
        max_steps: 每回合最大步数
        
    返回:
        path: 规划的路径点列表
        time_cost: 算法执行时间
        nodes_explored: 探索的节点数量
        visited_nodes: 访问过的节点列表
    """
    start_time = time.time()
    nodes_explored = 0
    visited_nodes = []
    
    # 定义状态和动作空间
    state_dim = 4  # (当前x, 当前y, 目标x, 目标y)
    action_dim = 4  # 上、下、左、右
    
    # 创建DQN智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=20000,
        batch_size=128
    )
    
    # 动作映射
    actions = [(0, -20), (0, 20), (-20, 0), (20, 0)]  # 上、下、左、右
    
    best_path = None
    best_reward = float('-inf')
    
    # 创建可视化器
    visualizer = DQNVisualizer()
    
    # 训练DQN
    for episode in range(max_episodes):
        current_pos = start
        path = [current_pos]
        episode_reward = 0
        steps_taken = 0
        episode_loss = None
        
        # 构建初始状态
        state = np.array([current_pos[0], current_pos[1], goal[0], goal[1]])
        
        for step in range(max_steps):
            # 选择动作
            action = agent.choose_action(state)
            dx, dy = actions[action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # 检查是否越界或碰到障碍物
            if (0 <= next_pos[0] < map_data.shape[0] and 
                0 <= next_pos[1] < map_data.shape[1] and 
                map_data[next_pos] > 0.5):
                current_pos = next_pos
            else:
                # 碰到障碍物或边界，保持原位
                next_pos = current_pos
            
            # 记录访问的节点
            if current_pos not in visited_nodes:
                visited_nodes.append(current_pos)
                nodes_explored += 1
            
            # 添加到路径
            if current_pos not in path:
                path.append(current_pos)
            
            # 计算奖励
            if current_pos == goal:
                reward = 100000  # 到达目标的奖励
                done = True
            elif next_pos == current_pos and next_pos != goal:
                reward = -10  # 碰到障碍物或边界的惩罚
                done = False
            else:
                reward = -1  # 每步的小惩罚
                done = False
            
            # 构建下一个状态
            next_state = np.array([current_pos[0], current_pos[1], goal[0], goal[1]])
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 学习
            loss = agent.learn()
            if loss is not None:
                episode_loss = loss
            
            state = next_state
            episode_reward += reward
            steps_taken = step + 1
            
            # 如果到达目标，结束当前回合
            if current_pos == goal:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_path = path.copy()
                break
        
        # 更新可视化
        visualizer.update(episode, episode_reward, agent.epsilon, episode_loss, steps_taken,
                         current_pos=current_pos, map_data=map_data, start=start, goal=goal)
        
        # 更新探索率
        agent.update_epsilon()
        
        # 如果已经找到路径并且训练足够多轮，可以提前结束
        if best_path and episode > 100 and episode % 50 == 0:
            if agent.epsilon < 0.1:  # 探索率已经很低
                break
    
    # 如果没有找到路径，返回None
    if not best_path or best_path[-1] != goal:
        visualizer.save()
        visualizer.close()
        return None, time.time() - start_time, nodes_explored, visited_nodes
    
    # 保存训练结果图表
    visualizer.save()
    visualizer.close()
    return best_path, time.time() - start_time, nodes_explored, visited_nodes

def dqn(map_data, start, goal, max_episodes=2000, max_steps=1000):
    """使用DQN算法进行路径规划的包装函数
    
    参数:
        map_data: 地图数据，0表示障碍物，1表示可通行区域
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)
        max_episodes: 最大训练回合数
        max_steps: 每回合最大步数
        
    返回:
        path: 规划的路径点列表
        time_cost: 算法执行时间
        nodes_explored: 探索的节点数量
        visited_nodes: 访问过的节点列表
    """
    return train_dqn(map_data, start, goal, max_episodes, max_steps)