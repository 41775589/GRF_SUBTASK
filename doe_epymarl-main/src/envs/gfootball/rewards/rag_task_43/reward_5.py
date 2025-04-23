import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that implements a reward function to enhance defensive gameplay
       which includes maintaining good positioning, intercepting passes, and transitioning to counter-attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_positioning": [0.0] * len(reward), "interceptions": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward based on defensive positioning
            # Closer to defensive line (x < 0 for left team, x > 0 for right team), higher reward
            if o['ball_owned_team'] == 1:  # if right team owns the ball
                if o['left_team'][o['active']][0] < -0.5:  # improved positioning for defensive left team
                    components["defensive_positioning"][rew_index] = 0.1
            elif o['ball_owned_team'] == 0:
                if o['right_team'][o['active']][0] > 0.5:  # improved positioning for defensive right team
                    components["defensive_positioning"][rew_index] = 0.1
            
            # Reward for interceptions
            if o['ball_owned_team'] != -1 and o['ball_owned_team'] != o['ball_owned_player']:
                if np.linalg.norm(o['ball'] - o['right_team'][o['active']]) < 0.1 or np.linalg.norm(o['ball'] - o['left_team'][o['active']]) < 0.1:
                    components["interceptions"][rew_index] = 0.2
                    
            # Compiling the reward with additional components
            reward[rew_index] += components["defensive_positioning"][rew_index]
            reward[rew_index] += components["interceptions"][rew_index]

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
