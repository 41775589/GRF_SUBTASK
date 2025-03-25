import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward based on defensive actions such as interceptions 
    and position based upon the proximity to the ball when the opposing team is in possession.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.interception_reward = 0.1
        self.positional_reward_weight = 0.05
        self.near_ball_threshold = 0.2  # Threshold to check nearness to the ball
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), 
                      "interception_reward": 0.0, 
                      "positional_reward": 0.0}
        observation = self.env.unwrapped.observation()
        for i, obs in enumerate(observation):
            ball_owned_team = obs['ball_owned_team']
            active_player_position = obs['left_team'][obs['active']]
            ball_position = obs['ball'][:2]

            # Enhance reward for successful interceptions
            if not obs['ball_owned_team'] == 0:  # Ball not owned by our team
                ball_previous_own_team = self.env.unwrapped.prev_state['ball_owned_team']
                if ball_previous_own_team == 1 and ball_owned_team != 1:
                    components["interception_reward"] += self.interception_reward

            # Additional reward based on defensive positioning when opponent possesses the ball
            if ball_owned_team == 1:
                distance_to_ball = np.linalg.norm(ball_position - active_player_position)
                if distance_to_ball < self.near_ball_threshold:
                    components['positional_reward'] += self.positional_reward_weight / distance_to_ball
            
            reward[i] += components["interception_reward"] + components["positional_reward"]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['interception_reward'] = self.interception_reward
        state['positional_reward_weight'] = self.positional_reward_weight
        state['near_ball_threshold'] = self.near_ball_threshold
        return state

    def set_state(self, state):
        self.interception_reward = state['interception_reward']
        self.positional_reward_weight = state['positional_reward_weight']
        self.near_ball_threshold = state['near_ball_threshold']
        self.env.set_state(state)
