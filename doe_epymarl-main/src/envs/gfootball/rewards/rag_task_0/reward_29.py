import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds new checkpoint rewards based on offensive football strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.offensive_checkpoints = [0.25, 0.5, 0.75, 1.0]  # Progressive checkpoints closer to opponent's goal
        self.checkpoint_rewards = [0.1, 0.15, 0.2, 0.25]  # Increments in reward for reaching each checkpoint
        self.checkpoints_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoints_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        components = {"base_score_reward": reward.copy(),
                      "offensive_strategy_reward": [0.0] * len(reward)}

        for index in range(len(reward)):
            player_obs = observation[index]
            position = player_obs['ball'] if 'ball' in player_obs else None
            if position is not None and player_obs['ball_owned_team'] == 1:  # if right team (offensive) has the ball
                distance_to_goal = 1 - position[0]  # since right goal is at x = 1
                for checkpoint, checkpoint_reward in zip(self.offensive_checkpoints, self.checkpoint_rewards):
                    if distance_to_goal <= checkpoint and index not in self.checkpoints_collected:
                        reward[index] += checkpoint_reward
                        components["offensive_strategy_reward"][index] += checkpoint_reward
                        self.checkpoints_collected[index] = True

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
