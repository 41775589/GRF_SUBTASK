import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for executing high passes with precision in football."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._pass_accuracy_checkpoints = 5
        self._distance_reward_increment = 0.1
        self._collected_checkpoints = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_precision_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['ball_owned_team'] != 1 or o['ball_owned_player'] != o['active']:
                continue

            if o['game_mode'] in [2, 4]:  # Focus on GoalKick and Corner modes
                ball_pos = o['ball']
                goal_pos = [1, 0]  # Simulating right goal at (1,0)
                distance_to_goal = np.linalg.norm(np.array(ball_pos[:2]) - np.array(goal_pos))

                # Reward based on distance decreased towards goal with the ball
                distance_score = max(0, 1 - distance_to_goal / 1.5)  # Normalize based on expected field length
                checkpoint_idx = int(distance_score * self._pass_accuracy_checkpoints)
                if checkpoint_idx > self._collected_checkpoints.get(rew_index, 0):
                    components["high_pass_precision_reward"][rew_index] = self._distance_reward_increment
                    reward[rew_index] += components["high_pass_precision_reward"][rew_index]
                    self._collected_checkpoints[rew_index] = checkpoint_idx

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
