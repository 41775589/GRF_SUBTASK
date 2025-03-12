import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive strategies, shooting accuracy, dribbling, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shoot_reward = 1.0
        self.pass_reward = 0.5
        self.dribble_reward = 0.3
        self.control_factor = 0.2
        
    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        state_pickle = self.env.get_state(to_pickle)
        return state_pickle

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shoot_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for i, rew in enumerate(reward):
            obs = observation[i]
            if obs['ball_owned_team'] == 0:  # ball is owned by the team of the agent
                player_id = obs['active']
                if obs['ball_owned_player'] == player_id:
                    # Calculate distance to opponent goal
                    goal_y = 0  # Assuming goal is centered at y = 0
                    dist_to_goal = abs(obs['ball'][0] - 1)  # Assuming right goal is at x = 1
                    components["shoot_reward"][i] = self.shoot_reward * np.exp(-dist_to_goal)
                    
                    # Checking dribble action effectiveness
                    if obs['sticky_actions'][8] == 1:  # Action dribble is active
                        components["dribble_reward"][i] = self.dribble_reward
                    
                    # Detect passes
                    if obs['sticky_actions'][5] == 1:  # action_long_pass
                        components["pass_reward"][i] = self.pass_reward
                    elif obs['sticky_actions'][6] == 1:  # action_high_pass
                        components["pass_reward"][i] = self.pass_reward / 2  # less reward for high passes
                    
            # Aggregate all rewards
            reward[i] += components["shoot_reward"][i] + components["dribble_reward"][i] + components["pass_reward"][i]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        return observation, reward, done, info
