import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on possession changes, favoring well-timed defensive and offensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_change_counter = 0
        self.last_possession_team = None
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_change_counter = 0
        self.last_possession_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['possession_change_counter'] = self.possession_change_counter
        to_pickle['last_possession_team'] = self.last_possession_team
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.possession_change_counter = from_pickle['possession_change_counter']
        self.last_possession_team = from_pickle['last_possession_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            current_possession_team = o.get('ball_owned_team')
            
            # Reward for possession change
            if current_possession_team != self.last_possession_team and current_possession_team != -1:
                self.possession_change_counter += 1
                components['possession_change_reward'][rew_index] = 1.0  # Reward for successful possession change
                reward[rew_index] += components['possession_change_reward'][rew_index]
                
            self.last_possession_team = current_possession_team

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
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[action]
        return observation, reward, done, info
