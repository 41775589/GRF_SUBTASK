import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward to promote dribbling maneuvers combined with dynamic positioning,
    which facilitates fluid transitions between defense and offense.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._dribble_start_reward = 0.1
        self._dribble_end_reward = 0.1
        self._dynamic_positioning_reward = 0.2

    def reset(self):
        """
        Reset the environment and the counters tracking the sticky actions.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the internal state within the pickle dictionary.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Load the internal state from the pickle dictionary.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward based on the current environment observations, focusing on dribbling and positioning.

        Parameters:
        reward (list[float]) - The current rewards from the environment for each agent.

        Returns:
        tuple (list[float], dict): A tuple containing the modified reward list and a debug dictionary with reward components.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            # Reward for starting or stopping dribble
            if o['sticky_actions'][9] == 1 and self.sticky_actions_counter[9] == 0:
                reward[rew_index] += self._dribble_start_reward
                components["dribble_reward"][rew_index] += self._dribble_start_reward
                self.sticky_actions_counter[9] += 1

            elif o['sticky_actions'][9] == 0 and self.sticky_actions_counter[9] == 1:
                reward[rew_index] += self._dribble_end_reward
                components["dribble_reward"][rew_index] += self._dribble_end_reward
                self.sticky_actions_counter[9] = 0
            
            # Reward for moving dynamically by checking player and ball positions to promote offensive and defensive transitions
            if o['active'] != -1:
                player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
                ball_pos = o['ball'][:2]
                distance = np.linalg.norm(player_pos - ball_pos)
                if distance < 0.2:
                    reward[rew_index] += self._dynamic_positioning_reward
                    components["positioning_reward"][rew_index] += self._dynamic_positioning_reward

        return reward, components

    def step(self, action):
        """
        Execute an action and update counters and observation.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action
        return observation, reward, done, info
