import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive coordination and transition reward, focusing on unified responses and ball distribution."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_checkpoints = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['TransitionRewardWrapper'] = self.transition_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.transition_checkpoints = from_pickle['TransitionRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward defensive actions such as interceptions and blocks
            if o['game_mode'] in [3, 4]:  # FreeKick or Corner
                additional_reward = 0.1
                components["transition_reward"][rew_index] += additional_reward
                reward[rew_index] += 1.1 * additional_reward

            # Reward successful transitions from defense to attack, identified by ball possession status changes
            if o['ball_owned_team'] == 0:  # Note: 0 denotes left team, adjust according to team
                if self.transition_checkpoints.get(rew_index, -1) != 0:
                    additional_reward = 0.2
                    components["transition_reward"][rew_index] += additional_reward
                    reward[rew_index] += 1.2 * additional_reward
                    self.transition_checkpoints[rew_index] = 0

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
