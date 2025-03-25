import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for efficient counter-attack initiation after recovering possession."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_change_points = 0.5  # Additional reward for changing possession
        self.counter_attack_points = 0.3  # Additional reward for moving the ball quickly towards the opponent after possession gain

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if 'ball_owned_team' in o and o['ball_owned_team'] != -1:
                previous_owner = self.env.unwrapped.get_state().get('previous_owner', -1)

                # Check for possession change
                if previous_owner != o['ball_owned_team']:
                    components["possession_change_reward"][rew_index] = self.possession_change_points
                    reward[rew_index] += components["possession_change_reward"][rew_index]
                    self.env.unwrapped.get_state()['previous_owner'] = o['ball_owned_team']

                # Reward moving the ball forward rapidly after possession change
                if o['ball_direction'][0] * (2 * o['ball_owned_team'] - 1) > 0.1:  # fast forward movement
                    components["counter_attack_reward"][rew_index] = self.counter_attack_points
                    reward[rew_index] += components["counter_attack_reward"][rew_index]

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
