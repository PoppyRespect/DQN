import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000  # 最大记忆容量
batch_size = 32  # 小批量随机梯度下降中的样本数量


class ReplayBuffer():
    """
    对Momery进行Put、Sample、Size等操作，形成一个Memery Buffer 方便使用
    """
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    """
    输出预测Q：即 Qnet 对 action的预测
    """
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        """
        :param obs: 观测变量
        :param epsilon: greedy算法中增加随机性的因子
        :return:
        """
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        """
        优化10次， 每个取batch_size 个数据
        """
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())  # 初始化网络的权重
    memory = ReplayBuffer()

    print_interval = 20  # 打印周期
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        #  训练 10000次游戏
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # 线性退化 从8% - 1% 感性理解就是越往后训练越保守，很少探索新的可能
        s = env.reset()  # 对环境初始化，返回的是一个state
        done = False  # 游戏是否结束的flag

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)  # 预测State下的Action
            s_prime, r, done, info = env.step(a)  # 进行下一步动作：返回下一时间刻的S_prime, S_prime->S的奖励，done，和info（多用于调试）
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))  # 增加记忆
            s = s_prime  # 更新状态

            score += r  # 更新分数
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)  # 训练模型

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0  # 分数归零
    env.close()


if __name__ == '__main__':
    main()
