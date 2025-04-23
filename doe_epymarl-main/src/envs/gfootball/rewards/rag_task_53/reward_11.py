import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds strategic checkpoint rewards focused on ball control and play distribution under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_control_rewards = {}
        self._ball_movement_reward = 0.05
        self._space_exploitation_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_control_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_control_rewards'] = self._ball_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._ball_control_rewards = from_pickle['ball_control_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_control_reward": [0.0] * len(reward),
            "space_exploitation_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_idx = o['active']

            # Analyze the distance from ball to active players and opponents to give control reward
            player_position = o['right_team'][player_idx] if o['ball_owned_team'] == 1 else o['left_team'][player_idx]
            ball_position = np.array(o['ball'][:2]) # Ball position (x, y)
            own_team_positions = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
            opponent_team_positions = o['left_team'] if o['ball_owned_team'] == 1 else o['right_team']

            nearest_opponent_distance = np.min(np.linalg.norm(opponent_team_positions - player_position, axis=1))
            nearest_teammate_distance = np.min(np.linalg.norm(own_team_positions - player_position, axis=1))

            # Reward for maintaining control under pressure
            if nearest_opponent_distance < 0.1:  # Close proximity to opponents
                components["ball_control_reward"][rew_index] += self._ball_movement_reward

            # Reward for distribution and exploiting space
            if nearest_teammate_distance > 0.3:  # If teammates are well-distributed
                components["space_exploitation_reward"][rew_index] += self._space_exploitation_reward

            # Update rewards based on components
            reward[rew_index] += components["ball_control_reward"][rew_index] + components["space_exploitation_reward"][rew_index]
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
