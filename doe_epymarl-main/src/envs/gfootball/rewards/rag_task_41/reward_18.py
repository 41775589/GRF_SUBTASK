import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on attacking strategies, including playmaking and finishing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "creative_play_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Iterate through each agent's observations.
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage moving towards the opponent's goal with the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Calculate how close the ball is to the opponent's goal on x-axis
                goal_distance = 1 - o['ball'][0]  # x-coordinates range from -1 to 1, goal is at x=1
                components["creative_play_reward"][rew_index] = goal_distance
                reward[rew_index] += 2 * goal_distance

            # Check if a goal is scored
            if o['score'][0] > o['score'][1]:  # Assuming increment in left_team's score
                components["creative_play_reward"][rew_index] += 5.0
                reward[rew_index] += 5.0

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Track actions that remain active using sticky actions
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
