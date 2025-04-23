import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to incentivize wide midfield responsibilities, including positioning and high passing to expand the field of play.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pass_reward = 0.2
        self._positioning_reward = 0.15
        self._expansion_effectiveness = 0.1

    def reset(self):
        """
        Reset sticky actions counter on each episode start.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save additional states and return the complete state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Retrieve the sticky actions counter and set the environment state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Adjust the rewards based on specific midfield wide tasks such as high passes and effective positioning.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "expansion_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for executing a high pass in a wide midfield scenario
            if 'sticky_actions' in o and o['sticky_actions'][7]:  # Assuming index 7 corresponds to 'action_high_pass'
                components["high_pass_reward"][rew_index] = self._high_pass_reward

            # Calculating reward for effective positioning on the wide midfield
            if ('left_team' in o and 'right_team' in o):
                # Check if the player is positioned in the wide areas of the field (approximately lateral 1/3rd of the field on either side)
                if abs(o['active'][1]) > 0.25:  # y-coordinate threshold for lateral wide positioning
                    components["positioning_reward"][rew_index] = self._positioning_reward

                # Reward based on the expansion effectiveness (player spreading the game)
                if ('right_team_direction' in o and o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']):
                    change_in_x = o['right_team_direction'][o['active']][0]  # Change in the x-coordinate (horizontal expansion)
                    if change_in_x > 0:
                        components["expansion_reward"][rew_index] = self._expansion_effectiveness * change_in_x

            reward[rew_index] += sum(components.values())
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_value  # Maintain count of sticky actions
        
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
