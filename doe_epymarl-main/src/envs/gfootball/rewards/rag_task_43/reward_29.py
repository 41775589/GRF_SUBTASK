import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward function to emphasize defensive skills.
    It introduces rewards for interception, good positioning, and counter-attack starts.
    """
    def __init__(self, env):
        super().__init__(env)
        self.interception_reward = 0.3
        self.good_positioning_reward = 0.2
        self.counter_attack_start_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Customize the reward function to improve defensive strategy by adding rewards
        for interception, good positioning relative to the ball and initiating counter-attacks.
        """
        observation = self.env.unwrapped.observation()
        updated_rewards = np.array(reward)
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "good_positioning_reward": [0.0] * len(reward),
            "counter_attack_start_reward": [0.0] * len(reward)
        }

        if observation is None:
            return updated_rewards, components

        for idx in range(len(reward)):
            obs = observation[idx]

            if obs['ball_owned_team'] == 1 and obs['active'] == obs['ball_owned_player']:
                # Rewards the player for intercepting the ball from the opposing team
                components["interception_reward"][idx] = self.interception_reward
    
            if obs['ball_owned_team'] == 0 and np.linalg.norm(obs['ball'][:2] - obs['right_team'][obs['active']]) < 0.1:
                # Reward the player for being in a good defensive position close to the ball
                components["good_positioning_reward"][idx] = self.good_positioning_reward

            if obs['game_mode'] == 6 and obs['ball_owned_team'] == 0:
                # Rewards starting a counter-attack if in possession after retrieving the ball from a dangerous position
                components["counter_attack_start_reward"][idx] = self.counter_attack_start_reward

            total_additional_reward = components["interception_reward"][idx] + \
                                      components["good_positioning_reward"][idx] + \
                                      components["counter_attack_start_reward"][idx]
            updated_rewards[idx] += total_additional_reward

        return updated_rewards, components

    def step(self, action):
        """
        Takes in an action and returns the new observation, altered reward, done signal and info.
        It does not change the behavior but wraps the reward returned by the env.
        """
        observation, reward, done, info = self.env.step(action)
        modified_reward, reward_components = self.reward(reward)

        info['final_reward'] = sum(modified_reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)

        return observation, modified_reward, done, info
