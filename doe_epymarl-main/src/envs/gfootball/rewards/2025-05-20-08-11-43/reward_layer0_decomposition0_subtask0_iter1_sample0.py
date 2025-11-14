import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for coordinated defensive actions and bolstering transitions during counter-attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "defense_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Encourages maintaining positions essential for a solid defense
            if o['left_team_active'][o['active']]:
                if o['ball_owned_team'] == 1:  # If the ball is owned by the opponent
                    distance = np.linalg.norm(o['left_team'][o['active']] - o['ball'])
                    if distance < 0.3:  # Close to the ball
                        components['defense_reward'][rew_index] += 1.0 - distance
                        reward[rew_index] += components['defense_reward'][rew_index]

            # Enhance rewards for successful progression from defending to advancing the ball
            if o['left_team_active'][o['active']] and o['ball_owned_team'] == 0:
                distance = np.linalg.norm(o['left_team'][o['active']] - [1, 0])  # Distance to opponent goal
                components['transition_reward'][rew_index] = max(0, 1 - distance)
                reward[rew_index] += components['transition_reward'][rew_index]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
