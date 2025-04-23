import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for sprinting and crossing abilities for wingers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # for tracking sticky actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "wing_cross_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reaction)
        }

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            # Determine if the controlled player is a winger (roles 6 or 7, which are midfielders)
            if obs['active'] != -1 and obs['right_team_roles'][obs['active']] in [6, 7]:
                # Crossing reward when the ball is near the sides of the field
                if abs(obs['ball'][1]) > 0.3:  # y position of the ball on the field
                    components["wing_cross_reward"][rew_index] = 0.5
                
                # Sprint reward when sprint action (index 8 for 'sprint' in sticky_actions) is active
                if obs['sticky_actions'][8] == 1:
                    components["sprint_reward"][rew_index] = 0.2

            # Compile all rewards for the current player
            total_reward = reward[rew_index] + components["wing_cross_reward"][rew_index] + components["sprint_reward"][rew_index]
            reward[rew_index] = total_reward

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
            for i, act in enumerate(agent_obs['sticky_actions']):
                if act:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
