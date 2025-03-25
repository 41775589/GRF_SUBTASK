import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on rewarding a sweeper for clearing the ball and defensive actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 1.0
        self.tackle_reward = 1.5
        self.last_ball_position = None
        self.own_goal_location = np.array([-1, 0])  # Assuming own goal at left side (-1, y)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'CheckpointRewardWrapper' in from_pickle:
            self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            obs = observation[i]
            ball_position = obs['ball'][:2]
            ball_owned_team = obs['ball_owned_team']
            player_position = obs['right_team'][obs['active']] if ball_owned_team == 1 else obs['left_team'][obs['active']]

            # Reward for clearing the ball from near own goal towards midfield or opponent's half
            if self.last_ball_position is not None:
                if np.linalg.norm(self.own_goal_location - ball_position) > np.linalg.norm(self.own_goal_location - self.last_ball_position):
                    if obs['ball_owned_team'] == 0:  # Sweeper's team owns the ball
                        components['clearance_reward'][i] = self.clearance_reward
                        reward[i] += components['clearance_reward'][i]
            
            # Reward for tackling if ball possession changes near own goal
            if obs['ball_owned_team'] == 1 and self.last_ball_position is not None:
                if np.linalg.norm(self.own_goal_location - player_position) < 0.3:
                    if self.last_ball_position == -1:  # Opponent had the ball in the last step
                        components['tackle_reward'][i] = self.tackle_reward
                        reward[i] += components['tackle_reward'][i]

            self.last_ball_position = ball_owned_team

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
