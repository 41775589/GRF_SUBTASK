import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on defense and midfield control."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_midfield_checkpoints = {}
        self.num_midfield_checkpoints = 5
        self.midfield_checkpoint_reward = 0.1
        self.defensive_actions_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_midfield_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.collected_midfield_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_midfield_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "midfield_checkpoint_reward": [0.0] * len(reward),
            "defensive_actions_reward": [0.0] * len(reward)
        }
        if not observation:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            midfield_y_positions = np.linspace(-0.42, 0.42, self.num_midfield_checkpoints)
            player_y_pos = o['right_team'][o['active']][1]  # y position of active player

            # Reward for maintaining positions in midfield and controlling the game
            closest_midfield_idx = np.argmin(np.abs(midfield_y_positions - player_y_pos))
            if self.collected_midfield_checkpoints.get(rew_index, -1) != closest_midfield_idx:
                components["midfield_checkpoint_reward"][rew_index] = self.midfield_checkpoint_reward
                reward[rew_index] += components["midfield_checkpoint_reward"][rew_index]
                self.collected_midfield_checkpoints[rew_index] = closest_midfield_idx

            # Reward for defensive actions: tackling and blocking
            if o.get('game_mode') in [3, 4]:  # FreeKick, Corner
                components["defensive_actions_reward"][rew_index] = self.defensive_actions_reward
                reward[rew_index] += components["defensive_actions_reward"][rew_index]

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
