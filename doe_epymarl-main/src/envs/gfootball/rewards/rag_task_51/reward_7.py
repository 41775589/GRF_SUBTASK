import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for specialized goalkeeper training focusing on shot-stopping, quick reflexes, and initiating counter-attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_saves = 0
        self.counter_attack_starts = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_saves = 0
        self.counter_attack_starts = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_saves'] = self.ball_saves
        to_pickle['counter_attack_starts'] = self.counter_attack_starts
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_saves = from_pickle['ball_saves']
        self.counter_attack_starts = from_pickle['counter_attack_starts']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "save_reward": [0.0] * len(reward),
            "counter_attack_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Check if our agent is the goalkeeper
            if o['active'] != 0 or o['left_team_roles'][o['active']] != 0:
                continue
            
            # Check if a save is made.
            if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball'] - o['left_team'][o['active']]) < 0.1:
                self.ball_saves += 1
                components['save_reward'][rew_index] = 1.0

            # Checking for counter-attack initiation
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'] and np.abs(o['ball_direction'][0]) > 0.3:
                self.counter_attack_starts += 1
                components['counter_attack_reward'][rew_index] = 0.5

            # Calculate combined reward
            reward[rew_index] += components['save_reward'][rew_index] + components['counter_attack_reward'][rew_index]

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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
