import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for shooting accuracy and power."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shot_power_reward": [0.0] * len(reward),
            "shot_accuracy_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            ball_owned_by_player = o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']
            game_mode_normal = o['game_mode'] == 0
            
            # Check if the player is in a position to shoot and the game is in normal mode
            if ball_owned_by_player and game_mode_normal:
                # Encourage shooting by measuring power (focusing on fast ball speed)
                components["shot_power_reward"][i] = np.clip(np.linalg.norm(o['ball_direction']), 0, 1)

                # Encourage accuracy by rewarding proximity to the goal when shooting
                distance_to_goal_y = abs(o['ball'][1])
                components["shot_accuracy_reward"][i] = 1.0 - distance_to_goal_y

                reward[i] += (components["shot_power_reward"][i] + components["shot_accuracy_reward"][i]) * 0.5

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
