import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering close-range attacks, specifically shot precision and dribble effectiveness against goalkeepers."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._precision_checkpoints = 5
        self._dribble_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "dribble_reward": [0.0] * len(reward),
                      "precision_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][:2]
            goal_pos = [1, 0]  # Simulating the goal position at right midline of field

            # Dribble reward condition
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                # Check if dribbling (action_id = 9)
                if o['sticky_actions'][9] == 1:
                    components["dribble_reward"][rew_index] = self._dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]

            # Precision reward condition
            distance_to_goal = np.linalg.norm(np.array(ball_pos) - np.array(goal_pos))
            if distance_to_goal < 0.1:  # arbitrary distance indicating close range
                components["precision_reward"][rew_index] = 1.0 - distance_to_goal
                reward[rew_index] += components["precision_reward"][rew_index]

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
