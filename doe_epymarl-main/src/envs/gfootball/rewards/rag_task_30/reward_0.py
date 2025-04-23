import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for tactical positioning and transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reset_counters()

    def reset_counters(self):
        self.defensive_positioning_counter = 0
        self.transition_counter = 0
        self.position_rewards = {}

    def reset(self):
        self.reset_counters()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_positioning_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if reward[rew_index] == 1:  # Score reward remains intact
                continue

            # Reward for maintaining a strategic defensive position
            if 'left_team_roles' in o and o['left_team_roles'][o['active']] in [1, 4, 5]:
                # Assuming roles 1, 4, and 5 are strategic defensive roles (CB, DM, CM)
                if (o['ball'][0] < -0.5) and (o['left_team'][o['active']][0] < -0.7):
                    self.defensive_positioning_counter += 1
                    if self.defensive_positioning_counter == 1:
                        components["defensive_positioning_reward"][rew_index] = 0.1

            # Reward for transitioning from defense to counterattack
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                if 'ball_direction' in o and o['ball_direction'][0] > 0:
                    self.transition_counter += 1
                    if self.transition_counter == 1:
                        components["transition_reward"][rew_index] = 0.2

            reward[rew_index] += (components["defensive_positioning_reward"][rew_index] +
                                  components["transition_reward"][rew_index])

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_data'] = {
            "defensive_positioning_counter": self.defensive_positioning_counter,
            "transition_counter": self.transition_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positioning_counter = from_pickle['CheckpointRewardWrapper_data']['defensive_positioning_counter']
        self.transition_counter = from_pickle['CheckpointRewardWrapper_data']['transition_counter']
        return from_pickle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, active_action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = active_action
        return observation, reward, done, info
