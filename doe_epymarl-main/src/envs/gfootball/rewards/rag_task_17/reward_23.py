import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on wide midfield strategies such as positioning
    and high passes to stretch the opposition's defense."""

    def __init__(self, env):
        super().__init__(env)
        self._num_checkpoints = 5
        self.pass_quality_reward = 0.1
        self.positioning_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        pickle_data = self.env.get_state(to_pickle)
        return pickle_data

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_quality_reward": np.zeros_like(reward),
            "positioning_reward": np.zeros_like(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['left_team'][o['active']]
            ball_position = o['ball'][:2]

            # Check if high pass is performed and successful
            if o['sticky_actions'][9] == 1:  # Assuming index 9 is high pass action
                pass_quality = np.linalg.norm(ball_position - active_player_pos)
                if o['ball_owned_team'] == 0:  # Ball is owned by left team after pass
                    components["pass_quality_reward"][rew_index] = self.pass_quality_reward * pass_quality
                    reward[rew_index] += components["pass_quality_reward"][rew_index]
            
            # Reward for good positioning on the field to stretch defense
            if active_player_pos[0] > 0 and abs(active_player_pos[1]) > 0.3:
                components["positioning_reward"][rew_index] = self.positioning_reward
                reward[rew_index] += components["positioning_reward"][rew_index]

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
