import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages mastering High Passing and wide midfield positioning, which aids in lateral transitions and stretches the opposition defense."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_threshold = 0.5  # Assume some means of quantifying pass quality, e.g., pass completion or appropriate teammate reception
        self.high_pass_action_index = 9  # Assuming index 9 corresponds to a High Pass action in the simulation environment
        self.reward_for_pass = 0.2
        self.reward_for_positioning = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Get latest observation from the environment
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward High Pass if the correct sticky action is out
            if o['sticky_actions'][self.high_pass_action_index] == 1:
                # Assuming that this check confirms a high pass was attempted
                components["high_pass_reward"][rew_index] = self.reward_for_pass
                reward[rew_index] += components["high_pass_reward"][rew_index]

            # Check if the agent is on the sides of the field (wide midfield positioning)
            agent_x, agent_y = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']] 
            if abs(agent_y) > 0.3:  # Y axis value to check for lateral positioning
                components["position_reward"][rew_index] = self.reward_for_positioning
                reward[rew_index] += components["position_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, original_reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(original_reward)
        info['final_reward'] = sum(modified_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, modified_reward, done, info
