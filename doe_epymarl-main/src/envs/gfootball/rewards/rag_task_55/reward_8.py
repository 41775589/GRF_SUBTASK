import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive actions, specifically focusing on tackles."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_success_bonus = 1.0
        self._no_foul_penalty = 0.5

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
                      "tackle_success_bonus": [0.0] * len(reward),
                      "no_foul_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if the current game mode is a defensive situation
            if o['game_mode'] in [3, 4, 5, 6]:  # Modes like FreeKick, Corner, ThrowIn, Penalty
                # If the player's team performed a tackle (without a card)
                if o['sticky_actions'][7] == 1:  # Action for "slide" is at index 7 in sticky actions
                    if not o['left_team_yellow_card'][o['active']] and not o['right_team_yellow_card'][o['active']]:
                        components["tackle_success_bonus"][rew_index] = self._tackle_success_bonus
                        reward[rew_index] += components["tackle_success_bonus"][rew_index]
                        # Add a no-foul penalty
                        components["no_foul_penalty"][rew_index] = self._no_foul_penalty
                        reward[rew_index] += components["no_foul_penalty"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
