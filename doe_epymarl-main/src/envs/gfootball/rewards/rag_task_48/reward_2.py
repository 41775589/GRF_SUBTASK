import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward in the Google Research Football environment
    to focus on executing high passes from midfield to create scoring opportunities.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Resets the environment and internal counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the internal state of the wrapper along with its environment's state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the internal state of the wrapper and the environment.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward based on successful high passes from midfield positions with the 
        potential to directly contribute to scoring opportunities.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        high_pass_coefficient = 0.5

        for i, o in enumerate(observation):
            ball_position = o['ball'][0]
            ball_direction = o['ball_direction']
            ball_owned_team = o['ball_owned_team']
            
            # High pass: ball must be moving upward (forward direction for the right team), with z > 0.1
            is_high_pass = ball_direction[2] > 0.1 and ball_owned_team == 1
            
            # Check if the player is in the midfield area (-0.2 < x < 0.2)
            in_midfield = -0.2 < ball_position < 0.2
            
            # Check if the ball is moving towards the scoring opportunity (forward direction and close to the goal area)
            toward_goal = ball_direction[0] > 0 and ball_position > 0.7
            
            if is_high_pass and in_midfield and toward_goal:
                components['high_pass_reward'][i] = high_pass_coefficient
                reward[i] += components['high_pass_reward'][i]
        
        return reward, components

    def step(self, action):
        """
        Steps through the environment, adjusting the reward according to the wrapper's specifications.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                
        return observation, reward, done, info
