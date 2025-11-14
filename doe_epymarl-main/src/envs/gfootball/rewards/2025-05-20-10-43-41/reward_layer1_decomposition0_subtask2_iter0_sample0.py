import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds a reward for defensive maneuvers in wide areas. """
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Counter to keep track of sticky actions for detailed interpretation later
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Resets the wrapper along with the environment. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """ Return state to possibly pickle and save the model state. """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set state as per the pickle provided. Adjust for this wrapping. """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """ Add rewards for defensive maneuvers: sliding tackling, intercept long balls, and using Stop-Dribble. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_skill_reward": [0.0]}

        # Early exit if observation is non-retrievable
        if observation is None:
            return reward, components
        
        # Transform raw reward into structured feedback
        assert len(reward) == 1  # Since the training involves only one agent.
        o = observation[0]
        defensive_skill_bonus = 0

        # Encourage intercepting long balls
        if o['ball_owned_team'] == -1 and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
            positional_diff = np.linalg.norm(o['ball'] - o['left_team'][o['active']])
            if positional_diff < 0.1:
                defensive_skill_bonus += 1.0
        
        # Reward for using sliding to tackle effectively
        if 'sliding' in o['sticky_actions']:
            defensive_skill_bonus += 0.5

        # Reward for halting dribbles effectively using Stop-Dribble
        if 'stop_dribble' in o['sticky_actions']:
            defensive_skill_bonus += 0.3

        # Record the calculated defensive skill bonus
        components["defensive_skill_reward"][0] = defensive_skill_bonus
        reward[0] += defensive_skill_bonus

        return reward, components

    def step(self, action):
        """ Processes environment steps, including custom reward adjustments. """
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
