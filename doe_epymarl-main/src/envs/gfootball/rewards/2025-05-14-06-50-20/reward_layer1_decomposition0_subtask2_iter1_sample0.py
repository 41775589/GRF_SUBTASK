import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on dribbling and sprint effectiveness to penetrate defenses."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_success_counter = 0
        self.sprint_efficiency_multiplier = 0.05

    def reset(self):
        """Resets the counters on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_success_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serializes the state including custom sticky actions state."""
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist(),
            'dribble_success_counter': self.dribble_success_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserializes the state including custom sticky actions state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        self.dribble_success_counter = from_pickle['CheckpointRewardWrapper']['dribble_success_counter']
        return from_pickle

    def reward(self, reward):
        """Enhances rewards based on dribbling and sprinting effectiveness."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward.copy()}

        components = {
            "base_score_reward": reward.copy(),
            "dribble_sprint_bonus": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sprint_active = o['sticky_actions'][8] == 1  # Index 8 relates to 'action_sprint'
            dribble_active = o['sticky_actions'][9] == 1  # Index 9 relates to 'action_dribble'

            if sprint_active:
                self.sticky_actions_counter[8] += 1

            if dribble_active:
                self.sticky_actions_counter[9] += 1

                # Increase dribble success count on successful dribble action
                if o['ball_owned_team'] == 1:
                    self.dribble_success_counter += 1

            efficiency_bonus = (self.dribble_success_counter * self.sprint_efficiency_multiplier +
                                self.sticky_actions_counter[8] * 0.02)
            components['dribble_sprint_bonus'][rew_index] += efficiency_bonus
            reward[rew_index] += efficiency_bonus

        return reward, components

    def step(self, action):
        """Applies the action, computes the rewards and returns results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
