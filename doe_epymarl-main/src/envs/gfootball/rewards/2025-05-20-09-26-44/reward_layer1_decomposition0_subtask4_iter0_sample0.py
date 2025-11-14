import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom reward wrapper focusing on defensive and quick transition subtasks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_checkpoint_reward_state'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['defensive_checkpoint_reward_state']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_actions_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = components["base_score_reward"][rew_index]
            defensive_score = components["defensive_actions_reward"][rew_index]

            # Assuming observations include boolean flags indicating successful defensive actions.
            successful_defense = o.get('successful_defense', False)
            successful_transition = o.get('successful_transition', False)

            if successful_defense:
                defensive_score += 0.5  # Reward for successful defense like sliding, stopping opponent's sprint
                self.sticky_actions_counter[rew_index] += 1

            if successful_transition:
                defensive_score += 0.3  # Additional reward for initiating counter-attacks or quick ball transitions

            reward[rew_index] = base_reward + defensive_score
            components["defensive_actions_reward"][rew_index] = defensive_score

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
