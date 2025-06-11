# --- 1. Define the Neural Network Architecture ---
class DQN(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) for Q-value approximation."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# --- 2. Define the Replay Buffer ---
class ReplayBuffer:
    """A fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save an experience."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences from memory."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

