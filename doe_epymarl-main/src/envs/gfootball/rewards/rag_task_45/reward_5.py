import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that augments the reward for mastering Stop-Sprint and Stop-Moving defensive moves.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.previous_speed = None

    def reset(self):
        """ Resets the reward wrapper state with the environment's reset. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.previous_speed = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Allows the state of the reward wrapper to be saved with the rest of the environment.
        """
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_ball_position': self.previous_ball_position,
            'previous_speed': self.previous_speed
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restores the reward wrapper state along with the environment state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.previous_ball_position = from_pickle['CheckpointRewardWrapper']['previous_ball_position']
        self.previous_speed = from_pickle['CheckpointRewardWrapper']['previous_speed']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function that alters the rewards based on the stop-sprint and stop-moving actions.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        components = {"base_score_reward": reward.copy()}

        for i, obs in enumerate(observation):
            if 'ball' in obs:
                current_ball_position = np.array(obs['ball'][:2])  # only consider x, y
                if self.previous_ball_position is not None:
                    current_speed = np.linalg.norm(current_ball_position - self.previous_ball_position)
                    if self.previous_speed is not None:
                        # Check if the player stopped moving or significantly reduced speed.
                        if current_speed < self.previous_speed * 0.1:
                            components["stop_reward"] = 0.2  # add constant reward for stopping
                            reward[i] += components["stop_reward"]
                        
                        # Check if the player began sprinting after stopping.
                        if obs['sticky_actions'][8] == 1 and self.previous_speed == 0:
                            components["sprint_reward"] = 0.3  # add constant reward for sprinting after stopping
                            reward[i] += components["sprint_reward"]

                self.previous_speed = current_speed
                self.previous_ball_position = current_ball_position

        return reward, components

    def step(self, action):
        """
        Environment interaction based on the specified action.
        """
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return obs, reward, done, info
