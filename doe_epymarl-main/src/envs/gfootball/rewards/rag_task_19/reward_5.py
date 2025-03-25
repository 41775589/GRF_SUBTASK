import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward based on defensive actions and midfield control."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defense_efficiency = 0.2  # incentive for effective defense
        self._midfield_control = 0.15   # incentive for controlling the midfield

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DefenseMidfieldReward'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('DefenseMidfieldReward', {}).get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_reward": [0.0] * len(reward),
                      "midfield_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Enhancing reward for defensive efficiency
            if o['game_mode'] in [2, 3, 4]:  # defensive situations: GoalKick, FreeKick, Corner
                if o['ball_owned_team'] == 0:  # own team has the ball
                    components["defense_reward"][rew_index] = self._defense_efficiency

            # Enhancing reward for controlled midfield play
            midfield_zone = (-0.3, 0.3)  # defining midfield along x-axis
            if midfield_zone[0] < o['ball'][0] < midfield_zone[1]:
                components["midfield_reward"][rew_index] = self._midfield_control

            reward[rew_index] += components["defense_reward"][rew_index] + components["midfield_reward"][rew_index]
        
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
