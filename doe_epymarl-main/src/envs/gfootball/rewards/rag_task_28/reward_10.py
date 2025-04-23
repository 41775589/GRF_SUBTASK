import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Adds a reward based on dribbling and performing feints under high-pressure scenarios."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize your specific variables here
        self._feint_reward = 0.1  # Reward increment for successfully dribbling past a defender
        self._pressure_factor = 0.05  # Additional reward if under pressure
        self._ball_control_bonus = 0.2  # Rewards for maintaining control

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # There may be data to retrieve here depending on whether you are saving state
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "feint_reward": [0.0] * len(reward),
                      "pressure_reward": [0.0] * len(reward),
                      "ball_control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the active player has the ball and is close to the opponent’s goal
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                # Calculate distance to the opponent's goal
                goal_distance = 1 - abs(o['ball'][0])  # goal is at x = ±1, ball[0] gives x-pos
                if goal_distance < 0.3:  # Arbitrary threshold for "close to goal"
                    components["feint_reward"][rew_index] = self._feint_reward * (1 - goal_distance)

                # Check for high pressure: close opponents
                min_distance = min(np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1))
                if min_distance < 0.1:  # Threshold for "under pressure"
                    components["pressure_reward"][rew_index] = self._pressure_factor * (0.1 - min_distance)

                # Reward for action "dribbling" if in possession and under pressure
                if o['sticky_actions'][9] == 1 and min_distance < 0.1:
                    components["ball_control_reward"][rew_index] = self._ball_control_bonus

            # Aggregate calculated rewards into the final reward vector
            reward[rew_index] += sum([components[key][rew_index] for key in components])
        
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
