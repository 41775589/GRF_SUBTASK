import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the defensive skills of agents by rewarding defensive positioning and effective counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = set()
        self.prepared_for_counterattack = set()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = set()
        self.prepared_for_counterattack = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', self.sticky_actions_counter)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_position_reward": [0.0] * len(reward),
                      "counterattack_setup_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for defending near the goal post, identifying the player's team through ball owned team
            if o['ball_owned_team'] == (rew_index % 2) and o['active'] in self.defensive_positions:
                components["defensive_position_reward"][rew_index] = 0.2
              
            # If the active player is within 0.2 distance to ball, prepared for counterattack
            if o['ball_owned_team'] == -1 and o['steps_left'] > 0 and np.linalg.norm(o['ball'][:2]) < 0.2:
                if o['active'] not in self.prepared_for_counterattack:
                    components["counterattack_setup_reward"][rew_index] = 0.3
                    self.prepared_for_counterattack.add(o['active'])

            reward[rew_index] += components["defensive_position_reward"][rew_index] + components["counterattack_setup_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Include each reward component in the information
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Monitor sticky actions for debugging
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
