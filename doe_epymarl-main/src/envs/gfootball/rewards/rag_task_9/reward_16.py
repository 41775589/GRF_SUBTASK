import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that applies a reward mechanism which encourages offensive gameplay. Specifically,
    it emphasizes actions such as Short Pass, Long Pass, Shot, Dribble, and Sprint toward creating scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.2
        self.shooting_reward = 0.3
        self.dribbling_reward = 0.1
        self.sprinting_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Adjusts the original rewards by giving additional points for offensive actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "sprinting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Penalize or reward based on sticky actions and ball possession.
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if o['sticky_actions'][7]:  # Short Pass
                    components["passing_reward"][rew_index] += self.passing_reward
                if o['sticky_actions'][9]:  # Long Pass
                    components["passing_reward"][rew_index] += self.passing_reward
                if o['sticky_actions'][8]:  # Shot
                    components["shooting_reward"][rew_index] += self.shooting_reward
                if o['sticky_actions'][9]:  # Dribble
                    components["dribbling_reward"][rew_index] += self.dribbling_reward
                if o['sticky_actions'][8]:  # Sprint
                    components["sprinting_reward"][rew_index] += self.sprinting_reward

            # Update rewards based on the components defined.
            reward[rew_index] += (
                components["passing_reward"][rew_index] +
                components["shooting_reward"][rew_index] +
                components["dribbling_reward"][rew_index] +
                components["sprinting_reward"][rew_index]
            )

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
