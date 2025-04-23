import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful tackles and ball recovery."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.success_tackles_counter = 0

    def reset(self):
        """Reset for new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.success_tackles_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the state."""
        to_pickle['CheckpointRewardWrapper'] = {
            'success_tackles_counter': self.success_tackles_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize the state."""
        from_pickle = self.env.set_state(state)
        self.success_tackles_counter = from_pickle['CheckpointRewardWrapper']['success_tackles_counter']
        return from_pickle

    def reward(self, reward):
        """Modify reward based on successful tackles and ball recovery."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_success_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] == 0:  # Normal Play
                controlled_player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]
                ball_owned_by_opponent = (o['ball_owned_team'] == 1)
                distance_to_ball = np.linalg.norm(controlled_player_pos - ball_pos)

                # If the opponent owns the ball and our player has recently tackled and recovered the ball
                if ball_owned_by_opponent and distance_to_ball < 0.015:
                    components["tackle_success_reward"][rew_index] = 1.0  # Reward for successful tackle and recovery
                    reward[rew_index] += components["tackle_success_reward"][rew_index]
                    self.success_tackles_counter += 1
            else:
                # No reward modification in set-pieces other than 'normal'
                continue

        return reward, components

    def step(self, action):
        """Step function."""
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
