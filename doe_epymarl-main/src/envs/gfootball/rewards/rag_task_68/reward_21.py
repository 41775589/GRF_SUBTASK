import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on offensive strategies like shooting, dribbling, and passing."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.shooting_reward = 0.5
        self.dribbling_reward = 0.1
        self.passing_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = None
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        state = self.env.set_state(from_pickle)
        return state

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage shooting close to the goal
            if o['game_mode'] in [3, 6]:  # FreeKick or Penalty
                components["shooting_reward"][rew_index] = self.shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]
                
            # Encourage dribbling: give rewards if performing dribbling in the presence of opponents
            if o['sticky_actions'][9]:  # action_dribble
                close_opponents = np.any([
                    np.linalg.norm(np.array(player) - o['ball'][:2]) < 0.1 for player in o['right_team']
                ])
                if close_opponents:
                    components["dribbling_reward"][rew_index] = self.dribbling_reward
                    reward[rew_index] += components["dribbling_reward"][rew_index]

            # Encourage innovative passes
            if o['sticky_actions'][0] or o['sticky_actions'][7]: # if action_left or action_bottom_left (implying potential long passes)
                components["passing_reward"][rew_index] = self.passing_reward
                reward[rew_index] += components["passing_reward"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
