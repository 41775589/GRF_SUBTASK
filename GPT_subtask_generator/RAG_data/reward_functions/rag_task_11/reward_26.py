import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A dense reward wrapper for enhancing offensive capabilities through fast-paced maneuvers and precision finishing."""

    def __init__(self, env):
        super().__init__(env)
        self._num_checkpoints = 10
        self._checkpoint_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # Add specific state details as needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Load specific state details as needed
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if o.get('ball_owned_team', -1) == 1:
                own_goals = o['score'][1]
                enemy_goals = o['score'][0]
            else:
                own_goals = o['score'][0]
                enemy_goals = o['score'][1]

            # Reward if closer to opponent’s goal
            x_ball, y_ball = o['ball'][:2]
            distance_to_goal = 1 - x_ball if o.get('ball_owned_team', -1) == 1 else x_ball + 1
            checkpoints_collected = int((1 - distance_to_goal) * self._num_checkpoints)
            components["checkpoint_reward"][i] = checkpoints_collected * self._checkpoint_reward
            
            # Encourages shots towards the goal
            if o['game_mode'] == 6 and own_goals > enemy_goals:
                components["checkpoint_reward"][i] += 0.5

            reward[i] += components["checkpoint_reward"][i]

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
