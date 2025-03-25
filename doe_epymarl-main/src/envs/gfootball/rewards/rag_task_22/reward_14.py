import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for sprint usage and covering more field defensively."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["sprint_positioning_reward"][rew_index] = 0

            if 'sticky_actions' in o:
                if o['sticky_actions'][8] == 1:
                    # Sprint action index in sticky actions is 8
                    self.sticky_actions_counter[rew_index] += 1

                    # Calculating how much ground covered this step while sprinting
                    # Both the left and right team could be controlled by a single agent.
                    # We assume here that the environment is about the left team.
                    player_pos = o['left_team'][o['active']]
                    player_vel = o['left_team_direction'][o['active']]
                    dist_covered = np.linalg.norm(player_vel) * 0.01  # Scaled down for reasonable reward

                    # Rewarding sprinting while player is closer to their own goal to encourage defensive sprint use
                    own_goal_pos = [-1, 0]  # Assuming the default left goal pos
                    dist_to_own_goal = np.linalg.norm(np.array(player_pos) - np.array(own_goal_pos))
                    if dist_to_own_goal > 0.5:  # Only reward when player is in defensive half
                        components["sprint_positioning_reward"][rew_index] = dist_covered

            reward[rew_index] += components["sprint_positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"reward_component_{key}"] = sum(value)
        
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
