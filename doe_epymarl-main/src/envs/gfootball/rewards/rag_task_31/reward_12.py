import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive plays such as tackles and slides, with bonuses for quick reaction to opponent's attacks."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_reward = 0.2
        self._slide_reward = 0.3
        self._quick_reaction_bonus = 0.1
        self._reaction_threshold = 5  # Number of steps considered as quick reaction

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Provides current observation
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "slide_reward": [0.0] * len(reward),
                      "quick_reaction_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            my_team = 'left_team' if rew_index == 0 else 'right_team'
            opponent_team = 'right_team' if rew_index == 0 else 'left_team'

            # Check if there is a tackle or slide action
            if o['sticky_actions'][6] or o['sticky_actions'][7]:  # Tackle(push) or slide
                components["tackle_reward"][rew_index] = self._tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]

            if o['sticky_actions'][9]:  # Slide
                components["slide_reward"][rew_index] = self._slide_reward
                reward[rew_index] += components["slide_reward"][rew_index]

            # Quick reaction bonus if reacting quickly to opponent's ball possession changes
            opponent_has_ball = o[opponent_team + '_roles'][o['active']] == o['ball_owned_player']
            if opponent_has_ball and self.sticky_actions_counter[rew_index] < self._reaction_threshold:
                components["quick_reaction_bonus"][rew_index] = self._quick_reaction_bonus
                reward[rew_index] += components["quick_reaction_bonus"][rew_index]

        self.sticky_actions_counter += 1
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Adding each component's sum to the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Reset sticky actions on each step to track quick reactions
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
