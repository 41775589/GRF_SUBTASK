import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on dribbling actions and dynamic positioning to facilitate transitions between defense and offense."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            dribbled = obs['sticky_actions'][9]  # the dribbling action index is 9
            
            # Check if dribbling has started or stopped
            if dribbled != self.sticky_actions_counter[rew_index]:
                components["dribbling_reward"][rew_index] = 0.2 * (1 if dribbled else -1)
                self.sticky_actions_counter[rew_index] = dribbled

            # Positioning reward based on movement towards opponent half when dribbling
            if dribbled and obs['ball_owned_team'] == 0:  # assuming 0 is the team index of the agent
                player_pos = obs['left_team'][obs['active']]
                opponent_goal_pos = 1  # assuming goal position is at x = 1
                distance_to_goal = opponent_goal_pos - player_pos[0]
                components["positioning_reward"][rew_index] = np.clip(distance_to_goal * 0.1, -0.1, 0.1)

            # Aggregate the rewards
            reward[rew_index] += components["dribbling_reward"][rew_index] + components["positioning_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Info updates for debugging and tracking purposes
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
