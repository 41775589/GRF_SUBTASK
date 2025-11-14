import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that introduces rewards for strategic positioning and transition emphasis."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Configuration: manage number of horizontal zones and vertical movement importance.
        self.num_horizontal_zones = 5
        self.horizontal_reward = 0.05
        self.reverse_step_penalty = -0.02
        self.sprint_reward = 0.01

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No state to restore as no persistent state is kept here
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "positional_reward": [0.0] * len(reward),
            "reverse_motion_penalty": [0.0] * len(reward),
            "sprint_bonus": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward based on X-axis position (only for active players)
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                field_zone = int(o['right_team'][o['active']][0] * self.num_horizontal_zones + self.num_horizontal_zones // 2)
                components["positional_reward"][rew_index] += field_zone * self.horizontal_reward

            # Penalize reverse movements
            if 'right_team_direction' in o:
                if o['right_team_direction'][o['active']][0] < 0:
                    components["reverse_motion_penalty"][rew_index] += self.reverse_step_penalty

            # Bonus for sprinting, checking sticky actions for sprint
            if o['sticky_actions'][8] == 1:  # Action index 8 corresponds to sprint
                components["sprint_bonus"][rew_index] += self.sprint_reward

            # Aggregating all rewards for each agent
            reward[rew_index] += (components["positional_reward"][rew_index] +
                                  components["reverse_motion_penalty"][rew_index] +
                                  components["sprint_bonus"][rew_index])

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