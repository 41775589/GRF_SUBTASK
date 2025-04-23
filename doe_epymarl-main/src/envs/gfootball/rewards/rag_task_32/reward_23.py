import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for wing players focusing on sprinting and accurate crossing from the wings."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.cross_threshold = 0.8  # Threshold to consider a close approach for crossing

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
                      "sprinting_reward": [0.0] * len(reward),
                      "crossing_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            agent_obs = observation[rew_index]
            player_x, player_y = agent_obs['right_team'][agent_obs['active']][:2] if agent_obs['ball_owned_team'] == 1 else agent_obs['left_team'][agent_obs['active']][:2]

            # Encourage sprinting down the wings
            sprinting_bonus = agent_obs['sticky_actions'][8]  # 8 is the index for the 'action_sprint'
            if sprinting_bonus:
                components['sprinting_reward'][rew_index] = 0.01
                reward[rew_index] += components['sprinting_reward'][rew_index]

            # Reward for making crosses from the wings near the opponent's goal line
            is_near_goal_line = abs(player_x) >= self.cross_threshold
            has_possession = agent_obs['ball_owned_team'] == (0 if player_x < 0 else 1)
            if is_near_goal_line and has_possession:
                components['crossing_reward'][rew_index] = 0.1
                reward[rew_index] += components['crossing_reward'][rew_index]

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
