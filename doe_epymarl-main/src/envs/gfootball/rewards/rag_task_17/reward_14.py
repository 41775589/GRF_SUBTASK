import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering high passes and wide midfield play."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_bonus_multiplier = 0.1
        self.position_bonus_multiplier = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.wide_positions = {
            'right_midfield': np.array([0.5, -0.42]),  # Right side near sideline
            'left_midfield': np.array([-0.5, 0.42])   # Left side near sideline
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_internal'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        internal_state = from_pickle.get('CheckpointRewardWrapper_internal', {})
        return from_pickle

    def reward(self, reward):
        observation = self.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_bonus": [0.0] * len(reward),
            "position_bonus": [0.0] * len(reward),
        }

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Bonus for successful high passes
            if o['ball_owned_team'] == 1 and o['sticky_actions'][9] == 1:  # Assuming index 9 corresponds to high pass
                components["pass_bonus"][i] = self.pass_bonus_multiplier
                reward[i] += components["pass_bonus"][i]

            # Bonus for positioning in wide areas of the midfield
            player_pos = np.array([o['right_team'][o['active']]])
            for pos_key, pos_val in self.wide_positions.items():
                if np.linalg.norm(player_pos - pos_val) < 0.1:  # close to the target positions
                    components["position_bonus"][i] = self.position_bonus_multiplier
                    reward[i] += components["position_bonus"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
