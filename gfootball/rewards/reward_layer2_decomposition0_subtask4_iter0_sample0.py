import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper specifically designed for the subtask of mastering the technique of sliding 
    to effectively manage tackles without foul play, emphasizing timing, precision, and minimizing penalties.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize any variables you may need here
        self.sliding_correctness_counter = np.zeros(1, dtype=int)
        self.precision_reward = 0.1

    def reset(self):
        self.sliding_correctness_counter = np.zeros(1, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        # Extract observations from environment to assess the situation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "precision_reward": [0.0]}

        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            # Checking if sliding action is conducted appropriately
            if 'sticky_actions' in obs and obs['sticky_actions'][9] == 1:  # Index 9 for sliding action
                if self._is_sliding_appropriate(obs):
                    components["precision_reward"][0] += self.precision_reward
                    reward[idx] += components["precision_reward"][0]
                    self.sliding_correctness_counter[idx] += 1
        
        return reward, components

    def _is_sliding_appropriate(self, observation):
        """Checks if the sliding made by the player is appropriate i.e., close to the opponent player."""
        ball_pos = observation.get('ball', [0, 0])
        player_pos = observation.get('right_team', [[0, 0]])[observation.get('active')]
        
        distance_to_ball = np.linalg.norm(np.array(ball_pos[:2]) - np.array(player_pos))
        # Check if the sliding is done at a close range to the ball which is likely close to the opponent
        return distance_to_ball < 0.1

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
