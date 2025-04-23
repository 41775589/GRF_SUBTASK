import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for controlling ball under pressure and strategic plays."""

    def __init__(self, env):
        super().__init__(env)
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
        new_rewards = reward.copy()  # Start with original rewards
        components = {
            "base_score_reward": reward.copy(),  # Original score rewards
            "ball_control_reward": [0.0] * len(reward),  # Reward for controlling the ball
            "strategic_play_reward": [0.0] * len(reward)  # Reward for strategic plays
        }

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            # Ball control under pressure
            if obs['ball_owned_team'] == 0 and np.any(obs['left_team_active']):
                components['ball_control_reward'][i] = 0.1
                new_rewards[i] += components['ball_control_reward'][i]

            # Strategic movements
            if (obs['ball_owned_team'] == 0 and obs['active'] == obs['ball_owned_player']):
                # Calculate distance to nearest opponent
                ball_pos = obs['ball'][:2]
                distances = np.linalg.norm(obs['right_team'] - ball_pos, axis=1)
                min_distance = np.min(distances)
                # Reward if ball is close to own player and far from opponents
                if min_distance > 0.2:  # Threshold for 'pressure'
                    components['strategic_play_reward'][i] += 0.2
                    new_rewards[i] += components['strategic_play_reward'][i]

        return new_rewards, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action

        return obs, reward, done, info
