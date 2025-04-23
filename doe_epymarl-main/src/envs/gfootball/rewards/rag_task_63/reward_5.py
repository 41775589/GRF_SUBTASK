import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for a goalkeeper mastering shot stopping, distribution under pressure, and communication."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_efficiency = 0.1  # Coefficient for goalkeeper efficiency

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.goalkeeper_efficiency
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_efficiency = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Add reward for successful shot stop attempts based on the goalie's actions
            if o['game_mode'] in [6]:  # Penalty kick game mode
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['designated']:
                    goalkeeper_action = self.extract_goalkeeper_action(o['sticky_actions'])
                    if goalkeeper_action in [0, 1, 2, 3, 4, 5, 6, 7]:  # Defensive/directional actions
                        components["goalkeeper_efficiency"] = [self.goalkeeper_efficiency] * len(reward)
                        reward[rew_index] += components["goalkeeper_efficiency"][rew_index]
        
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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def extract_goalkeeper_action(self, sticky_actions):
        # A hypothetical function to determine the goalkeeper's actions
        return np.argmax(sticky_actions)  # This assumes a method to decode goalkeeper's actions
