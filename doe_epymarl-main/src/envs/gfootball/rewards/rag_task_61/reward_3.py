import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies rewards focusing on team synergy during possession changes, 
    emphasizing strategic positioning and well-timed actions for both offensive and defensive shifts.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward tuning parameters
        self.position_reward_coefficient = 0.05
        self.synch_action_reward = 0.1
        
        # Track last positions to compare changes
        self.last_ball_position = None
        self.ball_change_over_time = None

    def reset(self):
        """
        Reset environment and clear counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.ball_change_over_time = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the current state of the environment alongside wrapper's variables.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['last_ball_position'] = self.last_ball_position
        to_pickle['ball_change_over_time'] = self.ball_change_over_time
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the environment state from pickle data.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.last_ball_position = from_pickle['last_ball_position']
        self.ball_change_over_time = from_pickle['ball_change_over_time']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward structure based on team coordination during a possession change.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "position_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Increment reward if there is strategic shift in positioning
            if self.last_ball_position is not None:
                dist = np.linalg.norm(self.last_ball_position - o['ball'])
                components['position_reward'][rew_index] = dist * self.position_reward_coefficient
                reward[rew_index] += components['position_reward'][rew_index]
            
            # Check for synchronized actions during possession change
            if o['ball_owned_team'] != self.env.unwrapped.observation()[rew_index]['ball_owned_team']:
                reward[rew_index] += self.synch_action_reward
            
        self.last_ball_position = observation[0]['ball']
        
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
