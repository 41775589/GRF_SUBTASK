import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a crossing and sprinting reward for wingers.
    Focuses on encouraging wingers to dribble to wing areas and produce crosses.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define distance thresholds and rewards for key areas (wing zones)
        self.crossing_reward = 1.0
        self.dribbling_reward = 0.1
        self.wing_thresholds = [0.8, -0.8]
        self.correct_game_mode = 0  # e_GameMode.Normal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_data'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['checkpoint_data']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.get_observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}

        assert len(reward) == len(observation)

        for idx, o in enumerate(observation):
            player_pos = o['right_team'][o['active']] if o['ball_owned_team'] == 1 else o['left_team'][o['active']]
            x, y = player_pos

            # Check if the player is in one of the wing zones and ready to cross
            if abs(y) > self.wing_thresholds[0] and o['game_mode'] == self.correct_game_mode:
                if 'cross' in o['sticky_actions']:  # Simplified assumption
                    components['crossing_reward'][idx] = self.crossing_reward

            # Reward dribbling in horizontal movement towards the wings
            if o['ball_owned_team'] in [0, 1] and 'sprint' in o['sticky_actions'] and 'dribble' in o['sticky_actions']:
                if abs(x) >= 0.5:  # Threshold to be considered close to wings
                    components['dribbling_reward'][idx] = self.dribbling_reward

            # Combine rewards
            reward[idx] += components['crossing_reward'][idx] + components['dribbling_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
