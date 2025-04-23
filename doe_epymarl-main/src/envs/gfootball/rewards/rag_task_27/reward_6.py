import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized defensive skill reward for mastering defensive responsiveness and interceptions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        # Modify the reward function based on defensive plays
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]

            # Increase reward for successful interception and defensive positioning
            if o['game_mode'] in [2, 4, 5] and o['ball_owned_team'] == 1:  # Defending corner kicks, throw-ins, free kicks
                components['defensive_reward'][rew_index] = 0.3
            if o['ball_owned_team'] == 0 and o['left_team_active'][o['ball_owned_player']]:
                # Higher reward for defending close to own goal
                distance_to_goal = abs(o['left_team'][o['ball_owned_player']][0] + 1)
                components['defensive_reward'][rew_index] = distance_to_goal * 0.2

            reward[rew_index] = base_reward + components['defensive_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        info['sticky_actions'] = self.sticky_actions_counter
        return observation, reward, done, info
