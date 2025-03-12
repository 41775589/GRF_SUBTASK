import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for offensive strategies including shooting, dribbling, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shoot_rewards = 0.3
        self.dribble_rewards = 0.2
        self.pass_rewards = 0.1
        self.attack_distance_threshold = 0.2  # distance to consider close to opponent's goal
        self.last_ball_position = np.array([0, 0])

    def reset(self, **kwargs):
        self.last_ball_position = np.array([0, 0])
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": np.zeros_like(reward),
                      "shoot_reward": np.zeros_like(reward),
                      "pass_reward": np.zeros_like(reward)}

        if observation is None:
            return reward, components

        for i in range(len(observation)):
            o = observation[i]
            ball_pos = np.array(o['ball'][:2])

            # Check if agent is close to the opponent's goal
            if ball_pos[0] > (1 - self.attack_distance_threshold):
                components["shoot_reward"][i] = self.shoot_rewards

            # Check if there is dribbling with the ball
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  
                # action_dribble is index 9
                components["dribble_reward"][i] = self.dribble_rewards

            # Check if there is a progressive pass
            if 'ball_direction' in o and np.linalg.norm(o['ball_direction'][:2] - self.last_ball_position) > 0.1:
                components["pass_reward"][i] = self.pass_rewards

            # Update last ball position
            self.last_ball_position = ball_pos

            # Update overall reward
            reward[i] += components["shoot_reward"][i]
            reward[i] += components["dribble_reward"][i]
            reward[i] += components["pass_reward"][i]

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)

        # Modify the reward using the reward() method
        reward, components = self.reward(reward)

        # Add final reward to the info
        info["final_reward"] = np.sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)

        return observation, reward, done, info
