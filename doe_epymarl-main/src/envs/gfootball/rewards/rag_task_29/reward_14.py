import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specifically enhances close-range shot precision and power adjustment."""

    def __init__(self, env):
        super().__init__(env)
        # A constant reward boost for successful precision shots
        self.precision_shot_reward = 1.0

        # Distance threshold for "close-range" scenario
        self.close_range_threshold = 0.2  # approximately 20% of the field length from the goal

        # Keeps track of sticky actions for debugging and additional features
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            # Check if we are shooting and close to the goal
            close_to_goal = abs(o['ball'][0]) > (1 - self.close_range_threshold)
            ball_owned_by_player = (o['ball_owned_team'] == (0 if o['left_team'][i] else 1))

            if close_to_goal and ball_owned_by_player:
                if o['game_mode'] == 6:  # game mode for shooting, considering penalty as a close range
                    reward[i] += self.precision_shot_reward
                    components.setdefault("precision_shot_reward", []).append(self.precision_shot_reward)
                else:
                    components.setdefault("precision_shot_reward", []).append(0)
            else:
                components.setdefault("precision_shot_reward", []).append(0)

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
