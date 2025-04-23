import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper focusing on technical skills for executing high passes with precision.
    This includes evaluating trajectory control, power assessment, and situational fruition for high passes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_quality_threshold = 0.5
        self.pass_power_coefficient = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Considers the action history of each agent

    def reset(self):
        """
        Reset the game environment and action counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save state with the checkpoint data.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Retrieve saved state and parse CheckpointRewardWrapper data.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Define the reward function enhancing the high pass execution skill.
        This is evaluated based on the trajectory quality, power, and utility of high passes.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_quality_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] in [3, 6]:  # Free kick or penalty scenarios
                ball_z = o['ball'][2]  # Z position of the ball to measure trajectory
                if ball_z > self.pass_quality_threshold:
                    # Evaluate power and control
                    power = np.linalg.norm(o['ball_direction'])
                    quality_reward = min(self.pass_power_coefficient * power, 1)
                    components["high_pass_quality_reward"][rew_index] = quality_reward
                    reward[rew_index] += quality_reward

        return reward, components

    def step(self, action):
        """
        Step function to pass on rewards after processing through the custom reward function.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            for i, sticky_action in enumerate(obs['sticky_actions']):
                if sticky_action:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
