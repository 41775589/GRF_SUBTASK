import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards precise long passes based on ball dynamics and accuracy."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward parameters
        self.long_pass_distance = 0.6  # Distance denoting a long pass
        self.accuracy_reward = 0.5  # Reward for accurate long pass
        self.completed_passes = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.completed_passes = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.completed_passes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.completed_passes = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "accuracy_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Track the starting and ending position of the ball for long passes
            if o['ball_owned_team'] == 1:  # If the right team owns the ball
                player_id = o['ball_owned_player']
                player_pos = o['right_team'][player_id]
                prev_ball_pos = o['ball'] - o['ball_direction']  # Approximate previous position
                
                if np.linalg.norm(prev_ball_pos - player_pos) >= self.long_pass_distance:
                    # Check accurate reception by any left team player
                    for other_player_pos in o['left_team']:
                        if np.linalg.norm(o['ball'] - other_player_pos) < 0.1:  # Threshold for catching the ball
                            # Reward for accurate long pass
                            if player_id not in self.completed_passes:
                                components["accuracy_bonus"][rew_index] = self.accuracy_reward
                                reward[rew_index] += components["accuracy_bonus"][rew_index]
                                self.completed_passes[player_id] = True
                                break

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
