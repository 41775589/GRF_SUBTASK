import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This wrapper enhances agent's offensive strategies by encouraging accurate shooting, 
    effective dribbling to evade opponents, and utilizing various pass types.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Constants to adjust weights for different rewarding strategies
        self.shooting_accuracy_reward = 0.3
        self.dribbling_skill_reward = 0.2
        self.passing_skill_reward = 0.3

    def reset(self):
        """ Reset the environment and the sticky actions counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Store state adjustments due to reward calculations."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Retrieve state adjustments due to reward calculations."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Modify the base reward based on offensive strategy performance,
        including shooting accuracy, dribbling, and passing.
        
        Args:
        reward (list[float]): Original rewards from the environment.
        
        Returns:
        tuple[list[float], dict[str, list[float]]]: Updated rewards and reward components.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward, copy=True),
                      "shooting_accuracy": [0.0] * 4,
                      "dribbling_skill": [0.0] * 4,
                      "passing_skill": [0.0] * 4}
                      
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for idx, o in enumerate(observation):
            # Increase reward for shots close to the goal
            if o['game_mode'] in [2, 6] and o['ball_owned_player'] == o['active']:  # GoalKick or Penalty
                components["shooting_accuracy"][idx] = self.shooting_accuracy_reward
                reward[idx] += components["shooting_accuracy"][idx]

            # Reward successful dribbling (evident from action 'dribble' being used)
            if o['sticky_actions'][9] == 1:  # 'dribble' action is active
                components["dribbling_skill"][idx] = self.dribbling_skill_reward
                reward[idx] += components["dribbling_skill"][idx]

            # Reward for using long and high passes (e.g., when mode changes to corner or throw in)
            if o['game_mode'] in [4, 5]:  # Corner or ThrowIn
                components["passing_skill"][idx] = self.passing_skill_reward
                reward[idx] += components["passing_skill"][idx]

        return reward, components

    def step(self, action):
        """
        Take a step using the given action and augment reward computation with details about reward composition.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
