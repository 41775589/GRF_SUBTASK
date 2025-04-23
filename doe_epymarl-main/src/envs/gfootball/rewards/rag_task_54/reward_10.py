import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on enhancing collaboration between passers and shooters."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._passing_checkpoints_collected = {}
        self._shooting_checkpoints_collected = {}
        self._passing_threshold = 0.3
        self._shooting_threshold = 0.1
        self._passing_reward = 0.05
        self._shooting_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Resetting the count of collected checkpoints for each episode
        self._passing_checkpoints_collected = {}
        self._shooting_checkpoints_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        # Include states of collected checkpoints in the pickle
        to_pickle['CheckpointRewardWrapper'] = {
            'passing_checkpoints': self._passing_checkpoints_collected,
            'shooting_checkpoints': self._shooting_checkpoints_collected
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Load the checkpoint states from pickle
        self._passing_checkpoints_collected = from_pickle['CheckpointRewardWrapper']['passing_checkpoints']
        self._shooting_checkpoints_collected = from_pickle['CheckpointRewardWrapper']['shooting_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "passing_reward": [0.0] * len(reward), 
                      "shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_id = o['active']
            ball_owner_team = o.get('ball_owned_team')
            ball_owner_player = o.get('ball_owned_player')
            player_pos = o['right_team'][player_id] if ball_owner_team == 1 else o['left_team'][player_id]

            # Check if the current player is near to making a successful pass.
            if player_id == ball_owner_player and ball_owner_team == 1:
                distances = np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1)
                if np.any(distances < self._passing_threshold):
                    # Reward for successful passes within range
                    components["passing_reward"][rew_index] = self._passing_reward
                    reward[rew_index] += components["passing_reward"][rew_index]
                    self._passing_checkpoints_collected[rew_index] = True

            # Check if the player is in a shooting position
            if player_id == ball_owner_player and np.abs(player_pos[0]) > 0.8:
                # Reward for taking a shot close to the goal
                components["shooting_reward"][rew_index] = self._shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]
                self._shooting_checkpoints_collected[rew_index] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Add components to the info dict for debugging purposes
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
