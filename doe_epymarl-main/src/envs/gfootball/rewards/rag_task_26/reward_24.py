import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances midfield play rewards in strategy-based football games."""

    def __init__(self, env):
        super().__init__(env)
        self.midfield_mark_rewards = {}
        self.num_midfield_zones = 5
        self.midfield_reward_increment = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_mark_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.midfield_mark_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_mark_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.copy(reward),
                      "midfield_play_reward": np.zeros(len(reward))}
        
        if observation is None:
            return reward, components

        for i, single_obs in enumerate(observation):
            # Check if in midfield zone and playing actively
            if 'left_team' in single_obs and 'right_team' in single_obs:
                # Identifying midfield zone from -0.3 to 0.3 in the x dimension
                midfield_mask = np.logical_and(single_obs['left_team'][:, 0] > -0.3,
                                               single_obs['left_team'][:, 0] < 0.3)
                
                if np.any(midfield_mask):
                    midfield_players = np.where(midfield_mask)[0]
                    # Reward for each midfield player marked, with diminishing returns
                    reward_increment = sum(self.midfield_reward_increment * 0.5**k
                                           for k, player in enumerate(midfield_players)
                                           if player not in self.midfield_mark_rewards.get(i, []))
                    components["midfield_play_reward"][i] = reward_increment
                    reward[i] += reward_increment

                    # Mark players to avoid repeated rewards
                    if i not in self.midfield_mark_rewards:
                        self.midfield_mark_rewards[i] = set()
                    self.midfield_mark_rewards[i].update(midfield_players)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
