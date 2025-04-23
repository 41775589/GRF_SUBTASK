import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds advanced offensive tactics rewards focusing on midfield and striker coordination."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and sticky action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        super().reset()
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the state by delegating to the environment's get_state method."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state by delegating to the environment's set_state method."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the reward based on advanced offensive play details."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "offensive_play_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active'] and obs['left_team_roles'][obs['ball_owned_player']] in (6, 8):  # midfielders
                # Check distance to striker and space creation
                striker_positions = [pos for idx, pos in enumerate(obs['left_team']) if obs['left_team_roles'][idx] == 9]  # strikers
                if striker_positions:
                    distance_to_strikers = np.min([np.linalg.norm(pos - obs['ball']) for pos in striker_positions])
                    # Reward closer distances within midfield possession
                    if distance_to_strikers < 0.2:
                        components['offensive_play_reward'][i] += 0.5

        # Combine base score and offensive play rewards
        for idx in range(len(reward)):
            reward[idx] += components['offensive_play_reward'][idx]

        return reward, components

    def step(self, action):
        """Perform an environment step and apply the reward wrap."""
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
