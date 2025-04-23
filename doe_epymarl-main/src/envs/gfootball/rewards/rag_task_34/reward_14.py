import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for mastering close-range attacks 
    by developing specialized skills in shot precision and dribble effectiveness 
    against goalkeepers, with emphasis on quick decision-making and agility.
    """

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
                      "close_range_attack_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]

            # Check if the controlled player is within a close range to the goal and has possession
            close_range_threshold = 0.3  # defines what is considered close range
            ball_x_position = o['ball'][0]

            # Checking if the controlled player has the ball and is in the opponent's close range
            if o['ball_owned_team'] == 1 and abs(ball_x_position) > (1 - close_range_threshold):
                # Encourage shots and dribbling when near the goal
                is_dribbling = o['sticky_actions'][9] == 1
                is_kicking = any(o['sticky_actions'][i] == 1 for i in range(10) if i != 9)  # all other actions considered as kick types

                # Weight importance of attack actions when close to goal
                if is_dribbling:
                    components["close_range_attack_reward"][rew_index] += 0.2
                if is_kicking:
                    components["close_range_attack_reward"][rew_index] += 0.5

            reward[rew_index] = base_reward + components["close_range_attack_reward"][rew_index]

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
