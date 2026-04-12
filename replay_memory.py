import sys
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.position = 0
        self.size = 0  # 現在の格納数

        # RNG
        self.rng = np.random.default_rng(seed)

        # 初期化（遅延確保）
        self.initialized = False

    def _initialize_buffers(self, state, action):
        state = np.asarray(state)
        action = np.asarray(action)

        self.state_dim = state.shape
        self.action_dim = action.shape if action.shape != () else (1,)

        # メモリ確保（float32で統一）
        self.state      = np.zeros((self.capacity, *self.state_dim), dtype=state.dtype)
        self.next_state = np.zeros((self.capacity, *self.state_dim), dtype=state.dtype)
        self.action     = np.zeros((self.capacity, *self.action_dim), dtype=action.dtype)
        self.reward     = np.zeros((self.capacity,))
        self.done       = np.zeros((self.capacity,))

        self.initialized = True

    def push(self, state, action, reward, next_state, done):
        # 初回のみサイズ確定
        if not self.initialized:
            self._initialize_buffers(state, action)

        idx = self.position

        # 直接代入（np.copy不要）
        self.state[idx]      = state
        self.next_state[idx] = next_state
        self.action[idx]     = action
        self.reward[idx]     = reward
        self.done[idx]       = done
        # print(state, next_state, action, reward, done)
        # print("--------------------------------------------------------------------------------------------------------------------")
        # print(self.state, self.next_state, self.action, self.reward, self.done)
        # sys.exit("エラーメッセージ: プログラムを終了します")

        # 更新
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(self.size, size=batch_size)

        state      = self.state[idx]
        action     = self.action[idx]
        reward     = self.reward[idx]
        next_state = self.next_state[idx]
        done       = self.done[idx]

        # print(state.shape, action.shape, reward.shape, next_state.shape, done.astype(np.float32).shape)
        # log_str = f"{state.shape}, {action.shape}, {reward.shape}, {next_state.shape}, {done.shape}\n"
        # with open("debug_shapes.txt", "a") as f:  # ← 追記モード
        #     f.write(log_str)
        # log_str = f"{state}, {action}, {reward}, {next_state}, {done}\n"
        # with open("debug_shapes.txt", "a") as f:  # ← 追記モード
        #     f.write(log_str)

        # sys.exit("エラーメッセージ: プログラムを終了します")

        return state, action, reward, next_state, done.astype(np.float32)

    def __len__(self):
        return self.size