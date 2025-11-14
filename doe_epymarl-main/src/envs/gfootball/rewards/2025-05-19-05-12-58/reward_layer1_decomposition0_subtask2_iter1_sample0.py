import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adjusts rewards to incentivize mastering Shooting and Dribbling, including control over Stop-Dribble and Sprint."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shot_attempts = 0
        self.dribbles_with_sprint = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.shot_attempts = 0
        self.dribbles_with_sprint = 0
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'shot_attempts': self.shot_attempts,
            'dribbles_with_sprint': self.dribbles_with_sprint
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shot_attempts = from_pickle['CheckpointRewardWrapper']['shot_attempts']
        self.dribbles_with_sprint = from_pickle['CheckpointRewardWrapper']['dribbles_with_sprint']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shot_reward": 0.0,
            "sprint_dribble_reward": 0.0
        }

        if observation is None:
            return reward, components
        
        o = observation[0]  # Assuming single agent in control

        # Shooting mechanics
        if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:  # ball possession by controlled player
            distance_to_goal = np.abs(o['ball'][0] - 1.0)  # approximate distance to opponent's goal from ball's x-position
            if distance_to_goal < 0.3:
                self.shot_attempts += 1
                components['shot_reward'] = 5.0 * np.exp(-distance_to_goal)  # more reward closer to the goal
        
        # Sprint and Dribble mechanics
        if o['sticky_actions'][9] == 1 and o['sticky_actions'][8] == 1:  # Dribble and Sprint active
            self.dribbles_with_sprint += 1
            components['sprint_dribble_reward'] = 0.2  # constant reward for sprint dribbling
        
        # Sum up rewards
        reward += components['shot_reward'] + components['sprint_dribble_reward']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
