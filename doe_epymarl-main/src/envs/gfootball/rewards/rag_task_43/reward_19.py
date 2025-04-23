import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward signals for defensive plays and transitions to counterattacks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.zeros(3)  # X, Y, Z position of the ball
        self.position_checkpoint = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Position checkpoints
        self.checkpoint_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.zeros(3)
        self.checkpoint_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['previous_ball_position'] = self.previous_ball_position
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['previous_ball_position']
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        self.update_checkpoints(observation, reward, components)

        return reward, components

    def update_checkpoints(self, observation, reward, components):
        current_ball_position = np.array(observation['ball'][0:2])  # We ignore Z for simplicity

        movement_distance = np.linalg.norm(self.previous_ball_position - current_ball_position)
        self.previous_ball_position = current_ball_position

        reward_contribution = 0.0
        # Calculate the reward based on the ball advancing towards opponent's goal
        if movement_distance > 0:  # Assuming non-zero movement
            for checkpoint in self.position_checkpoint:
                if self.checkpoint_rewards.get(checkpoint, True):
                    if current_ball_position[0] > checkpoint and self.previous_ball_position[0] <= checkpoint:
                        reward_contribution += 0.1  # Incremental reward for crossing a checkpoint
                        self.checkpoint_rewards[checkpoint] = False  # Mark checkpoint as collected

        reward[0] += reward_contribution  # Update the reward for the main player
        if 'checkpoint_reward' not in components:
            components['checkpoint_reward'] = [0.0] * len(reward)
        components['checkpoint_reward'][0] += reward_contribution
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
