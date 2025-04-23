import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for developing offensive strategies,
    focusing on accurate shooting, effective dribbling, and different types of passes.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward parameters
        self.shooting_reward = 0.5
        self.dribbling_reward = 0.3
        self.pass_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward),
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Ball possession and shot towards goal
            if o['ball_owned_team'] == 1 and o['ball'][0] > 0.5:
                components["shooting_reward"][rew_index] += self.shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]

            # Dribbling - effective when moving towards opponent's side with the ball
            if o['ball_owned_team'] == 1 and self.sticky_actions_counter[9] > 0:  # dribble action is active
                if np.linalg.norm(o['ball_direction'][:2]) > 0:  # ball is moving
                    components["dribbling_reward"][rew_index] += self.dribbling_reward
                    reward[rew_index] += components["dribbling_reward"][rew_index]

            # Pass effectiveness, focusing on long and high passes
            if self.sticky_actions_counter[3] > 0 or self.sticky_actions_counter[4] > 0:  # right or left passes
                if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball_direction'][:2]) > 0.3:  # significant ball movement
                    components["pass_reward"][rew_index] += self.pass_reward
                    reward[rew_index] += components["pass_reward"][rew_index]

            # Update sticky actions counter based on actions
            sticky_actions = o.get('sticky_actions', [])
            for i in range(10):
                if sticky_actions[i] == 1:
                    self.sticky_actions_counter[i] += 1
                else:
                    self.sticky_actions_counter[i] = 0

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Adding the components to the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update info with sticky action counts
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
