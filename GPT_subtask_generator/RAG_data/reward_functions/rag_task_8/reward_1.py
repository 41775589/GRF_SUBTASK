import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on quick ball recovery and initiating counter-attacks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._quick_recovery_reward = 1.0
        self._counter_attack_bonus = 1.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "quick_recovery_reward": [0.0] * len(reward),
                      "counter_attack_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for quick ball possession recovery
            if o['ball_owned_team'] == 1 and self._previous_ball_owned_team != 1:
                components["quick_recovery_reward"][rew_index] = self._quick_recovery_reward
                reward[rew_index] += self._quick_recovery_reward

            # Bonus for successful counter-attack initiation
            if o['game_mode'] in [1, 2, 3, 4, 5, 6]:  # Modes indicating ball recovery
                if np.linalg.norm(o['ball_direction']) > 0.5:  # Assuming initiation motion
                    components["counter_attack_bonus"][rew_index] = self._counter_attack_bonus
                    reward[rew_index] += self._counter_attack_bonus

        self._previous_ball_owned_team = o['ball_owned_team']
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
