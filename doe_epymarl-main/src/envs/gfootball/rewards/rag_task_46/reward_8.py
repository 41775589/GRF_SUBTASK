import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance the learning of standing tackles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_coefficient = 2.0  # Adjust this coefficient to scale the reward of successful tackles
        self.penalty_coefficient = -5.0  # Penalty for risky tackles leading to foul

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
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for agent_idx in range(len(reward)):
            o = observation[agent_idx]
            components["tackle_reward"][agent_idx] = 0

            # Check if there was a recent tackle attempt based on sticky actions
            if o['sticky_actions'][3] == 1 or o['sticky_actions'][4] == 1:  # Assuming these indices are for tackle actions
                # Check for successful tackle: if the ball possession changes with tackle action
                if o['game_mode'] in [0, 2] and o['ball_owned_team'] == 0:  # Normal play or set piece defense
                    components["tackle_reward"][agent_idx] = self.tackle_success_coefficient * (1 if o['ball_owned_player'] == o['active'] else 0)
                elif o['game_mode'] in [1, 3, 4, 5, 6] and o['left_team_yellow_card'][o['active']]:
                    # Consider any game mode where a foul might happen
                    components["tackle_reward"][agent_idx] = self.penalty_coefficient

            # Total reward for the agent
            reward[agent_idx] = components["base_score_reward"][agent_idx] + components["tackle_reward"][agent_idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
