import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a passing under pressure reward based on controlled 
    passes, ball possession during tight situations and progression towards the opponent's goal.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions for further analysis

    def reset(self):
        """ Reset the environment and the sticky action counters. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Save the current internal state of this wrapper. """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restore the internal state of this wrapper. """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """ Calculate the additional rewards for advanced ball control and passing under pressure. """
        # Observations from the environment provided to the reward function
        observation = self.env.unwrapped.observation()

        components = {'base_score_reward': reward.copy(),  # Base game reward (goal, out, etc.)
                      'control_passing_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] in (2, 3, 4, 5):  # Tight game situations like free kicks, corners, throw-ins
                if o['ball_owned_team'] == 0:  # Our team has the ball
                    components['control_passing_reward'][rew_index] += 0.1  # Encourage maintaining possession
                    
                    # Incentivise forward passes
                    if 'ball_direction' in o and o['ball'][0] > 0:
                        components['control_passing_reward'][rew_index] += 0.05 * o['ball'][0]

            # Increment reward based on calculated components
            reward[rew_index] += components['control_passing_reward'][rew_index]

        return reward, components

    def step(self, action):
        """ Overrides parent class method to inject reward modifications and gather sticky action stats. """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        
        # Gather component info for detailed statistics.
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter
        new_obs = self.env.unwrapped.observation()  # Fetch fresh observation after the step 
        if new_obs:
            for agent_obs in new_obs:
                self.sticky_actions_counter += agent_obs['sticky_actions']
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
