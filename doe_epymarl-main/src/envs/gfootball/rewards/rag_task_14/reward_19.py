import gym
import numpy as np
class SweeperRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that implements a specific reward function catering to the 'Sweeper' role in a soccer game.
    The role's focus is on clearing the ball from defensive zones, performing last-man tackles, and rapid recoveries.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state and include the SweeperRewardWrapper data for pickle save.
        """
        to_pickle['SweeperRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state from pickle loading including custom wrapper data if any.
        """
        from_pickle = self.env.set_state(state)
        # Handle any necessary state restoration here if state was saved.
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function based on the 'Sweeper' role's game play needs.
        
        Criteria for reward adjustments:
        - Increment when clearing the ball from own goal area.
        - Increment when performing a successful last-man tackle.
        - Increment when ball is quickly moved from own half to opponent's half.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        base_score_reward = reward.copy()
        new_rewards = [0.0] * len(reward)
        
        for idx, r in enumerate(reward):
            o = observation[idx]
            # Criteria 1: Ball clearing from defensive zone
            if o['ball_owned_team'] == (0 if 'left_team' in o else 1):
                if np.linalg.norm(o['ball']) < 0.2:  # Assuming close proximity to own goal
                    new_rewards[idx] += 0.1
            
            # Criteria 2: Last-man tackle
            # Hypothetical check if the last defender successfully tackles the ball carrier
            if o.get('last_man', False) and o['ball_owned_team'] == -1:
                new_rewards[idx] += 0.5
            
            # Criteria 3: Fast recovery and advancement to opponent's half
            # Check if team rapidly moves from defense to attack
            if o['ball_direction'][0] > 0.1:  # Hypothetical threshold for quick attack
                new_rewards[idx] += 0.2

        reward = [original + additional for original, additional in zip(base_score_reward, new_rewards)]
        
        components = {'base_score_reward': base_score_reward, 'sweeper_reward': new_rewards}
        return reward, components

    def step(self, action):
        """
        Execute one time step within the environment with the adjusted rewards.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_status in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_status
        return observation, reward, done, info
