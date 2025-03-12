import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive plays, encouraging accurate shooting,
    dribbling to evade opponents, and different types of passes."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_reward = 0.2
        self.shoot_reward = 0.3
        self.dribble_reward = 0.1
        self.rewards_per_player = {i: {} for i in range(env.action_space.n)}

    def reset(self):
        """Resets the environment and any internally tracked states."""
        self.rewards_per_player = {i: {} for i in range(self.env.action_space.n)}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serializes the internal state for the reward wrapper."""
        to_pickle['CheckpointRewardWrapper'] = self.rewards_per_player
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserializes the internal state for the reward wrapper."""
        from_pickle = self.env.set_state(state)
        self.rewards_per_player = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modifies the rewards based on the type of action performed by the player."""
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        assert len(reward) == len(observation), "Mismatch in reward and observation lengths."

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            pass_actions = {6, 7}  # Assuming 6, 7 are the ids for long and high pass actions
            shoot_action = 3       # Assuming 3 is the id for shoot action
            dribble_action = 9     # Assuming 9 is the dribble action

            active_actions = o['sticky_actions']
            if active_actions[shoot_action] == 1:
                components['shoot_reward'][rew_index] = self.shoot_reward
            if any(active_actions[action_id] == 1 for action_id in pass_actions):
                components['pass_reward'][rew_index] = self.pass_reward
            if active_actions[dribble_action] == 1:
                components['dribble_reward'][rew_index] = self.dribble_reward

            reward[rew_index] += (components['shoot_reward'][rew_index] +
                                  components['pass_reward'][rew_index] +
                                  components['dribble_reward'][rew_index])

        return reward, components

    def step(self, action):
        """Steps through the environment, modifies the reward, and adds reward components and final reward to info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
