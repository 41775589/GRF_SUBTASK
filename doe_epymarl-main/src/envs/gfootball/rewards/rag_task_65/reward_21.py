import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for shooting and passing activities in specific strategic contexts,
    aimed at promoting goal-oriented precision, decision-making, and player positioning during
    simulation training in a football game environment.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.shooting_zones = np.linspace(-1, 0, num=5)  # Five zones approaching enemy goal
        self.passing_zones = np.linspace(0, 1, num=5)  # Five zones approaching own goal
        self.shot_success_reward = 1.0
        self.pass_success_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shot_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward)
        }

        for o_index, o in enumerate(observation):
            # Increase reward for successful shots in different strategic zones near the goal
            if o['game_mode'] == 6 and o['ball'][0] >= self.shooting_zones[o['ball_owned_player']]:
                components["shot_reward"][o_index] = self.shot_success_reward

            # Increase reward for successful passes in different zones
            if o['ball_owned_team'] == 1 and o['ball'][0] <= self.passing_zones[o['ball_owned_player']]:
                components["pass_reward"][o_index] = self.pass_success_reward

        # Update the rewards with new components
        for i in range(len(reward)):
            reward[i] += components["shot_reward"][i] + components["pass_reward"][i]

        return reward, components

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
