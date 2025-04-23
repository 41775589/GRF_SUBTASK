import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for executing wide midfield roles effectively,
    particularly focusing on high pass execution, correct positioning, and facilitating lateral play.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define high pass and lateral movement thresholds
        self.high_pass_action = 7  # Assuming index 7 is high pass in sticky actions
        self.lateral_position_threshold = 0.3  # Y-position threshold for lateral expansion

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_counter']
        return from_pickle

    def reward(self, reward):
        # Initial components for the calculation of the reward
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0, 0.0],
                      "position_reward": [0.0, 0.0]}

        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            # Access each agent's observation
            o = observation[rew_index]

            # Reward for executing a high pass
            if o['sticky_actions'][self.high_pass_action]:
                components["high_pass_reward"][rew_index] = 0.5  # Assuming a high value for successful execution

            # Reward based on lateral positioning to promote field expansion
            player_y_position = o['left_team'][o['active']][1]  # Based on active player's Y position
            if abs(player_y_position) > self.lateral_position_threshold:
                components["position_reward"][rew_index] = 0.3  # Reward for good lateral positioning

            # Summarize all components to obtain the final reward for this step
            reward[rew_index] += (components["high_pass_reward"][rew_index] +
                                  components["position_reward"][rew_index])
        
        return reward, components

    def step(self, action):
        # Get the outputs from the original environment's step function
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Add modified reward to info for monitoring
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions counter for observation
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
