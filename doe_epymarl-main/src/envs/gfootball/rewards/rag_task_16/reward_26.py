import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for executing high precision passes accurately.
    This includes evaluating the accuracy and appropriateness of high passes
    (lob passes) within the game scenario. This reward incentivizes high passes
    that reach teammates in advantageous positions.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_reward = 0.2

    def reset(self):
        """
        Reset the environment and clear the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Enhance the reward based on the high pass execution.
        Each time a high lob pass effectively reaches a teammate, a reward is given.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "pass_accuracy_reward": [0.0, 0.0]}

        for rew_index, o in enumerate(observation):
            if o['game_mode'] in {2, 4}:  # Game modes that could involve high passes: Free Kicks, Corners
                if o['ball_owned_team'] == 1:  # If right team owns the ball
                    ball_end_position = o['ball'] + o['ball_direction'] * 3  # Approximate future position
                    teammates_positions = o['right_team']
                    
                    # Check if the ball will land near any teammate
                    for teammate_pos in teammates_positions:
                        distance = np.linalg.norm(teammate_pos - ball_end_position[:2])
                        
                        # Reward for passes that effectively reach within a small radius
                        if distance < 0.1:
                            components["pass_accuracy_reward"][rew_index] += self.pass_accuracy_reward
                            reward[rew_index] += components["pass_accuracy_reward"][rew_index]
                            break

        return reward, components

    def step(self, action):
        """
        Execute a step in the environment, compute the reward, and return the modified reward and observation.
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
