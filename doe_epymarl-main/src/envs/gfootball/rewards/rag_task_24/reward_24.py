import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward function to encourage mid to long-range passing in the football game."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize the counter for tracking the number of sticky actions for each agent
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define rewards for successful mid to long-range passing
        self.pass_success_reward = 0.5

    def reset(self):
        # Reset the sticky actions counter on a new episode
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Serialize the wrapper state for saving and resuming
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Deserialize the wrapper state for loading
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Early exit if ball is not owned or owned by the opposing team
            if o['ball_owned_team'] == -1 or o['ball_owned_team'] != self.unwrapped.player_role:
                continue

            # Calculate if the pass was long range by measuring movement of the ball
            ball_dist_change = np.linalg.norm(o['ball_direction'][:2])
            # Long-range passing threshold
            long_pass_threshold = 0.3

            # Check if the ball is moved by more than the threshold distance
            if ball_dist_change > long_pass_threshold:
                components["passing_reward"][rew_index] = self.pass_success_reward
                reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Execute environment step
        observation, reward, done, info = self.env.step(action)
        # Augment the reward using the custom reward function
        reward, components = self.reward(reward)
        # Save the summed reward to info
        info["final_reward"] = sum(reward)
        # Save components to info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions count
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
