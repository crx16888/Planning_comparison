import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class Node:
    def __init__(self, pos, g_cost=0, h_cost=0):
        self.pos = pos  # 节点位置 (x, y)
        self.g_cost = g_cost  # 从起点到当前节点的实际代价
        self.h_cost = h_cost  # 从当前节点到目标的估计代价
        self.f_cost = g_cost + h_cost  # 总代价
        self.parent = None  # 父节点，用于路径回溯
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class DQN_Network(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, action_dim=4):
        super(DQN_Network, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 使用全连接层替代图卷积层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        # 使用全连接层进行前向传播
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=20000, batch_size=128, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)
        self.learn_step_counter = 0
        
        # 创建在线网络和目标网络
        self.q_network = DQN_Network(state_dim, hidden_dim, action_dim)
        self.target_network = DQN_Network(state_dim, hidden_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    def choose_action(self, state):
        """选择动作，使用epsilon-greedy策略"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def learn(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 从经验回放中随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并更新Q网络
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 定期更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

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
    points = np.linspace([x1, y1], [x2, y2], num=20).astype(int)
    return all(map_data[x, y] > 0 for x, y in points)

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