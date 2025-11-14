import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds an auxiliary reward for the player focusing on shooting and dribbling techniques.
    This reward increases the player's performance in finishing and controlling the ball in the offensive zone.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # There are 10 sticky actions available
        self.shooting_reward_coef = 0.2
        self.dribbling_reward_coef = 0.1

    def reset(self):
        """
        Reset the reward wrapper state for a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the current state of the reward wrapper and environment for serialization.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the reward wrapper and environment from deserialization.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on the agent's performance in shooting and dribbling.

        Parameters:
        - reward (list of floats): The original rewards returned from the environment.

        Returns:
        - Modified rewards (list of floats)
        - Reward components (dict): Details of different components of the reward.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "shooting_reward": [0.0] * len(reward), "dribbling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                # If the active player owns the ball, incentivize shooting and dribbling
                if self.sticky_actions_counter[9] > 0:  # Dribbling action index is 9
                    components["dribbling_reward"][i] = self.dribbling_reward_coef * 1.5
                    reward[i] += components["dribbling_reward"][i]

                if self.sticky_actions_counter[6] > 0 or self.sticky_actions_counter[5] > 0:  # Shooting action indexes are 6 and 5
                    components["shooting_reward"][i] = self.shooting_reward_coef * 2.0
                    reward[i] += components["shooting_reward"][i]
        
        return reward, components

    def step(self, action):
        """
        Take an action in the environment and process the reward.

        Parameters:
        - action (var): The actions chosen by the environment's agents.

        Returns:
        - Observation, reward, done, info: Standard OpenAI gym step output.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        for agent_obs in obs:
            for idx, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[idx] = self.sticky_actions_counter[idx] + act if act == 1 else self.sticky_actions_counter[idx]

        return observation, reward, done, info
