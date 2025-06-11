import gymnasium as gym
# --- 4. Evaluation Function ---
def evaluate_and_record():
    """Load a trained model and record its performance as a GIF."""
    print("Starting evaluation and recording...")

    # --- Load the model ---
    input_dim = 4
    output_dim = 2
    policy_net = DQN(input_dim, output_dim)
    policy_net.load_state_dict(torch.load("dqn_cartpole_model.pth"))
    policy_net.eval()

    # --- Setup environment and recorder ---
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    frames = []

    # --- Run one episode ---
    obs, info = env.reset()
    done = False

    while not done:
        # Record the frame
        frame = env.render()
        frames.append(frame)

        # Choose action
        with torch.no_grad():
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy_net(state).argmax().item()

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()

    # --- Save the GIF ---
    imageio.mimsave('cartpole_demo.gif', frames, fps=30)
    print("Evaluation GIF saved to cartpole_demo.gif")


# --- 5. Main Execution ---
if __name__ == "__main__":
    train()
    evaluate_and_record()
