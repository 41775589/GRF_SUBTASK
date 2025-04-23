import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a midfield dynamics specialized reward."""

    def __init__(self, env):
        super().__init__(env)
        self.player_positions = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['MidfieldDynamicsRewardWrapper'] = self.player_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.player_positions = from_pickle['MidfieldDynamicsRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "midfield_dynamic_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["midfield_dynamic_reward"][rew_index] = 0

            # Manipulate reward based on midfield dominance and control:
            # Encourage players to maintain positions that bridge defense and attack effectively
            midfield_control_factor = 0.05

            if 'left_team' in o and 'right_team' in o:
                for player_pos in o['left_team']:
                    if -0.2 < player_pos[0] < 0.2:  # consider midfield region horizontally
                        components["midfield_dynamic_reward"][rew_index] += midfield_control_factor

                for player_pos in o['right_team']:
                    if -0.2 < player_pos[0] < 0.2:  # consider midfield region horizontally
                        components["midfield_dynamic_reward"][rew_index] += midfield_control_factor

            reward[rew_index] += components["midfield_dynamic_reward"][rew_index]
        
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
