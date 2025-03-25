import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages offensive strategies, optimizing team coordination and reaction, by rewarding passing, player positioning, and approaching the goal."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._pass_completion_reward = 0.1
        self._positioning_reward = 0.05
        self._approaching_goal_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_completion_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "approaching_goal_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward for pass completion
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and \
               'action' in o and o['action'] == 'pass':
                components["pass_completion_reward"][rew_index] = self._pass_completion_reward

            # Reward for good positioning
            if 'right_team_roles' in o:
                avg_y_pos = np.mean(o['right_team'][:, 1])
                components["positioning_reward"][rew_index] = self._positioning_reward if avg_y_pos < 0 else 0

            # Reward for approaching the goal
            if 'ball' in o:
                distance_to_goal = 1 - o['ball'][0]  # Assuming the goal is at x=1
                if distance_to_goal > 0:
                    components["approaching_goal_reward"][rew_index] = self._approaching_goal_reward * (1 - distance_to_goal)

            reward[rew_index] += sum(components[c][rew_index] for c in components)

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
