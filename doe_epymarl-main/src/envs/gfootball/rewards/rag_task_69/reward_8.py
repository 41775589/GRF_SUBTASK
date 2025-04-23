import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages mastering accurate shooting, effective dribbling, and varied passing strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.05
        self.dribbling_reward = 0.1
        self.shooting_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CustomState'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CustomState', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Initializing observation and components
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Evaluate each reward component based on current observations for each agent
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o['sticky_actions']

            # Check dribbling (Action 9 = dribbling action is activated)
            if sticky_actions[9]:
                components["dribbling_reward"][rew_index] = self.dribbling_reward
                reward[rew_index] += self.dribbling_reward
            
            # Evaluate accurate shooting towards goal based on ball possession and direction
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Simulating shot effectiveness by proximity to goal and orientation
                ball_pos = o['ball'][:2]
                goal_y_range = -0.044, 0.044
                if ball_pos[0] > 0.8 and goal_y_range[0] <= ball_pos[1] <= goal_y_range[1]:
                    components["shooting_reward"][rew_index] = self.shooting_reward
                    reward[rew_index] += self.shooting_reward

            # Evaluate passing strategies based on changes in ball direction and ownership
            # If the ball direction changes significantly while owned by player
            if np.linalg.norm(o['ball_direction'][:2]) > 0.5 and o['ball_owned_team'] == 0:
                components["passing_reward"][rew_index] = self.passing_reward
                reward[rew_index] += self.passing_reward

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
