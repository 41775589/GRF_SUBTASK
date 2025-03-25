import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for efficient stamina management using Stop-Sprint and Stop-Moving."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stamina_threshold = 0.3  # Threshold below which stamina management becomes crucial
        self.stamina_reward = 0.01   # Incremental reward for managing stamina.

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
                      "stamina_management_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_tired_factor = o['left_team_tired_factor' if o['active'] in o['left_team'] else 'right_team_tired_factor'][o['active']]
            
            # Check if the player is managing stamina below a certain threshold
            if player_tired_factor < self.stamina_threshold:
                if not o['sticky_actions'][8]:  # no sprint
                    components["stamina_management_reward"][rew_index] += self.stamina_reward
                    reward[rew_index] += components["stamina_management_reward"][rew_index]

                # Additional reward for stopping movement completely to regain stamina
                is_moving = np.any(o['sticky_actions'][:8])
                if not is_moving:
                    components["stamina_management_reward"][rew_index] += self.stamina_reward
                    reward[rew_index] += components["stamina_management_reward"][rew_index]

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
