import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards based on team synergy during possession changes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_changes = 0
        self.prev_ball_owner_team = None
        self.position_checkpoints = 5
        self.position_change_reward = 0.1

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.possession_changes = 0
        self.prev_ball_owner_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['possession_changes'] = self.possession_changes
        to_pickle['prev_ball_owner_team'] = self.prev_ball_owner_team
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.possession_changes = from_pickle['possession_changes']
        self.prev_ball_owner_team = from_pickle['prev_ball_owner_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        # Identify if there has been a possession change
        current_ball_owner_team = observation['ball_owned_team']
        if self.prev_ball_owner_team is not None and current_ball_owner_team != self.prev_ball_owner_team and current_ball_owner_team != -1:
            self.possession_changes += 1
            # Reward for recovering the ball
            components["possession_change_reward"] = [self.possession_changes * 0.5] * len(reward)

        self.prev_ball_owner_team = current_ball_owner_team

        # Reward based on team's strategic positioning during possession change
        for player in observation['left_team'] + observation['right_team']:
            if np.linalg.norm(player - observation['ball']) < 0.1:  # Player is close to the ball
                components["positioning_reward"] = [self.position_change_reward] * len(reward)

        combined_rewards = np.array(reward) + np.array(components["possession_change_reward"]) + np.array(components["positioning_reward"])
        return list(combined_rewards), components

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
