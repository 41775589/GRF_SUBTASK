import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards advanced ball control and passing under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_reward = 0.5  # Reward for successfully completing passes

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state to be pickled."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state from unpickled data."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Calculate rewards for advanced ball control and effective passing under pressure.
        This considers successful short, high, and long passes, increasing the reward when done under game pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Incentivize ball possession
            if o["ball_owned_team"] != 0:
                continue

            passing_types = [
                o["sticky_actions"][5],  # action_bottom_right represents High Pass
                o["sticky_actions"][6],  # action_bottom represents Long Pass
                o["sticky_actions"][7]   # action_bottom_left represents Short Pass
            ]
            success_factor = sum(passing_types)

            # Only reward if the player successfully passes under game conditions
            if o['ball_owned_player'] == o['active']:
                components["passing_reward"][rew_index] = self.pass_completion_reward * success_factor
                reward[rew_index] += 1.5 * components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Take a step in the environment and augment reward contributions."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add the rewards and components to info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Handling sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active

        return observation, reward, done, info
