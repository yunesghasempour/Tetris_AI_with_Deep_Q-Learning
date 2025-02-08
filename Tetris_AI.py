import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# مقداردهی اولیه pygame
pygame.init()

# تنظیمات بازی: برد 5 ستون در 10 ردیف
BLOCK_SIZE = 30
GRID_WIDTH = 5
GRID_HEIGHT = 10
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

# رنگ‌ها
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [
    (0, 255, 255),   # Cyan
    (255, 255, 0),   # Yellow
    (128, 0, 128),   # Purple
    (0, 255, 0),     # Green
    (255, 0, 0),     # Red
    (0, 0, 255),     # Blue
    (255, 127, 0)    # Orange
]

# تنها یک نوع قطعه: مربع 1x1
SHAPES = [
    [[1]]
]

# مدل DQN با معماری کانولوشنی (ورودی: تصویر وضعیت با ابعاد (1,10,5))
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # ورودی: 1 کانال، اندازه برد: GRID_HEIGHT×GRID_WIDTH = 10×5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  
        # خروجی conv1: (10-3+1=8) × (5-3+1=3) با 32 کانال
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # خروجی conv2: (8-3+1=6) × (3-3+1=1) با 64 کانال
        conv_output_size = 64 * 6 * 1  # =384
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_actions)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class TetrisGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris RL')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        self.board = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.current_piece = self._new_piece()
        self.game_over = False
        self.score = 0
        return self._get_state()
    
    def _new_piece(self):
        shape = np.array(random.choice(SHAPES))
        return {
            'shape': shape,
            'x': GRID_WIDTH // 2 - shape.shape[1] // 2,
            'y': 0,
            'color': random.randint(0, len(COLORS) - 1)
        }
    
    def _check_collision(self, shape, x, y):
        shape_height, shape_width = shape.shape
        if x < 0 or x + shape_width > GRID_WIDTH or y + shape_height > GRID_HEIGHT:
            return True
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i][j]:
                    if y + i >= 0 and self.board[y + i][x + j]:
                        return True
        return False
    
    def _merge_piece(self):
        shape = self.current_piece['shape']
        x = self.current_piece['x']
        y = self.current_piece['y']
        color_value = self.current_piece['color'] + 1
        shape_height, shape_width = shape.shape
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i][j]:
                    if 0 <= y + i < GRID_HEIGHT and 0 <= x + j < GRID_WIDTH:
                        self.board[y + i][x + j] = color_value
                        
    def _clear_lines(self):
        new_board = []
        lines_cleared = 0
        for row in self.board:
            if np.all(row != 0):
                lines_cleared += 1
            else:
                new_board.append(row)
        if lines_cleared > 0:
            empty_rows = np.zeros((lines_cleared, GRID_WIDTH))
            self.board = np.vstack((empty_rows, np.array(new_board)))
        return lines_cleared
    
    def _get_state(self):
        state = np.copy(self.board)
        piece = self.current_piece['shape']
        x = self.current_piece['x']
        y = self.current_piece['y']
        shape_height, shape_width = piece.shape
        for i in range(shape_height):
            for j in range(shape_width):
                if piece[i][j]:
                    if 0 <= y + i < GRID_HEIGHT and 0 <= x + j < GRID_WIDTH:
                        state[y + i][x + j] = 2
        return state[np.newaxis, :]
    
    def _rotate_piece(self, shape):
        return np.rot90(shape, -1)
    
    def step(self, action):
        """
        اعمال:
            0: چپ
            1: راست
            2: چرخش
            3: حرکت به سمت پایین
        """
        reward = 0
        x = self.current_piece['x']
        y = self.current_piece['y']
        shape = self.current_piece['shape']
        
        if action == 0:  # چپ
            if not self._check_collision(shape, x - 1, y):
                self.current_piece['x'] -= 1
        elif action == 1:  # راست
            if not self._check_collision(shape, x + 1, y):
                self.current_piece['x'] += 1
        elif action == 2:  # چرخش
            new_shape = self._rotate_piece(shape)
            if not self._check_collision(new_shape, x, y):
                self.current_piece['shape'] = new_shape
        elif action == 3:  # حرکت به سمت پایین
            if not self._check_collision(shape, x, y + 1):
                self.current_piece['y'] += 1
                # در حرکت پایین بدون برخورد، امتیازی تعلق نمی‌گیرد (scoring استاندارد فقط از پاکسازی ردیف‌ها است)
            else:
                self._merge_piece()
                lines = self._clear_lines()
                # امتیاز استاندارد: 1 ردیف = 100, 2 = 300, 3 = 500, 4 یا بیشتر = 800
                if lines == 1:
                    reward = 100
                elif lines == 2:
                    reward = 300
                elif lines == 3:
                    reward = 500
                elif lines >= 4:
                    reward = 800
                else:
                    reward = 0
                self.current_piece = self._new_piece()
                if self._check_collision(self.current_piece['shape'],
                                         self.current_piece['x'],
                                         self.current_piece['y']):
                    self.game_over = True
                    reward = -50
                self.score += reward
                return self._get_state(), reward, self.game_over
        
        # حرکت خودکار به سمت پایین (گرانش)
        new_x = self.current_piece['x']
        new_y = self.current_piece['y']
        new_shape = self.current_piece['shape']
        if not self._check_collision(new_shape, new_x, new_y + 1):
            self.current_piece['y'] += 1
        else:
            self._merge_piece()
            lines = self._clear_lines()
            if lines == 1:
                reward = 100
            elif lines == 2:
                reward = 300
            elif lines == 3:
                reward = 500
            elif lines >= 4:
                reward = 800
            else:
                reward = 0
            self.current_piece = self._new_piece()
            if self._check_collision(self.current_piece['shape'],
                                     self.current_piece['x'],
                                     self.current_piece['y']):
                self.game_over = True
                reward = -50
        self.score += reward
        return self._get_state(), reward, self.game_over
    
    def render(self):
        self.screen.fill(BLACK)
        # رسم بلوک‌های ثابت روی برد
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                if self.board[i][j]:
                    color_idx = int(self.board[i][j]) - 1
                    pygame.draw.rect(self.screen, COLORS[color_idx],
                                     (j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        # رسم قطعه جاری
        shape = self.current_piece['shape']
        color = COLORS[self.current_piece['color']]
        shape_height, shape_width = shape.shape
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i][j]:
                    pygame.draw.rect(self.screen, color,
                                     ((self.current_piece['x'] + j) * BLOCK_SIZE,
                                      (self.current_piece['y'] + i) * BLOCK_SIZE,
                                      BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        pygame.display.flip()
        self.clock.tick(30)

class TetrisAI:
    def __init__(self, num_actions):
        self.action_size = num_actions
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # نگه داشتن سطح بالای اکتشاف
        self.learning_rate = 0.001
        
        self.policy_net = DQN(num_actions)
        self.target_net = DQN(num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def get_action(self, state):
        # تبدیل وضعیت به تنسور با شکل (1,1,10,5)
        state = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_ai():
    env = TetrisGame()
    num_actions = 4  # اعمال: چپ، راست، چرخش، پایین
    agent = TetrisAI(num_actions)
    
    episodes = 1000
    target_update = 10
    
    try:
        for episode in range(episodes):
            state = env.reset()  # شکل: (1,10,5)
            total_reward = 0
            
            while not env.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train()
                total_reward += reward
                state = next_state
                env.render()
                
            if episode % target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Episode {episode+1}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        pygame.quit()

if __name__ == "__main__":
    train_ai()
