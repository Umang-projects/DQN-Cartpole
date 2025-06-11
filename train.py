# --- 3. Training Function ---
def train():
    """Main training loop."""
    # --- Hyperparameters ---
    learning_rate = 1e-3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    tau = 0.005  # For soft update of target network
    buffer_capacity = 10000
    num_episodes = 600

    # --- Initialization ---
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Set target network to evaluation mode

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)
    criterion = nn.MSELoss()

    all_rewards = []
    epsilon = epsilon_start

    print("Starting training...")
    for episode in range(num_episodes):
        observation, info = env.reset()
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()

            # Interact with the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Prepare tensors for the replay buffer
            action_tensor = torch.tensor([action], dtype=torch.int64)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            next_state = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0) if not done else None

            # Push to replay buffer
            replay_buffer.push(state, action_tensor, reward_tensor, next_state, done)

            state = next_state

            # --- Learning Step ---
            if len(replay_buffer) >= batch_size:
                # Sample a minibatch
                transitions = replay_buffer.sample(batch_size)

                # Unpack the batch
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                # Process the batch
                batch_state = torch.cat(batch_state)
                batch_action = torch.cat(batch_action)
                batch_reward = torch.cat(batch_reward)

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])

                # Get current Q-values from the policy network
                current_q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1))

                # Get next Q-values from the target network
                next_q_values = torch.zeros(batch_size)
                with torch.no_grad():
                    next_q_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

                # Compute the target Q-values
                target_q_values = batch_reward + (gamma * next_q_values)

                # Compute loss
                loss = criterion(current_q_values, target_q_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Soft update the target network
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = sum(all_rewards[-100:]) / 100
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    print("Training finished.")
    env.close()

    # --- Save the trained model ---
    torch.save(policy_net.state_dict(), "dqn_cartpole_model.pth")
    print("Model saved to dqn_cartpole_model.pth")

    # --- Plot the results ---
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards, label='Reward per Episode')
    # Calculate and plot a moving average
    moving_avg = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(99, len(all_rewards)), moving_avg, label='100-episode Moving Average', color='red')
    plt.title('Training Performance on CartPole-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("Learning curve saved to learning_curve.png")
    plt.show()
