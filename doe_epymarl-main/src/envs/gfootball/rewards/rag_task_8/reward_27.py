import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a special reward for quick decision-making and efficient ball handling to initiate counter-attacks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the state of the environment including sticky actions."""
        state = self.env.get_state(to_pickle)
        state['sticky_actions'] = self.sticky_actions_counter.tolist()
        return state

    def set_state(self, state):
        """Sets the state of the environment including the sticky actions."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions'])
        return from_pickle

    def reward(self, reward):
        """Modify the reward function to focus on quick possession changes and counter-attacks."""
        observations = self.env.unwrapped.observation()  # access to raw observations
        base_reward = reward.copy()
        
        counterattack_bonus = [0, 0]

        if observations is None:
            return reward, {}
        
        for i, observation in enumerate(observations):
            # We check if the ball changes possession to the controlled agent's team
            if observation['ball_owned_team'] == observation['active'] and self.sticky_actions_counter[9] == 1:
                # If ball is in possession and player is dribbling, give a bonus 
                distance_to_goal = np.abs(observation['ball'][0] - 1)  # normalize distance to opponent's goal line
                counterattack_bonus[i] = 2 * (1 - distance_to_goal)  # reward based on proximity to the goal

            reward[i] += counterattack_bonus[i]
        
        return reward, {'base_score_reward': base_reward, 'counterattack_bonus': counterattack_bonus}

    def step(self, action):
        """Perform a step using the given action, augment rewards using the custom reward function."""
        observation, reward, done, info = self.env.step(action)
        self.update_sticky_actions_counter(observation)
        
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info

    def update_sticky_actions_counter(self, observation):
        """Updates the count of sticky actions based on current observation."""
        for obs in observation:
            self.sticky_actions_counter = np.array(obs['sticky_actions'])
