import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on midfield dynamics, coordination under pressure, and strategic positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_position = 0.1  # Example boundary for midfield
        self.repositioning_rewards = {
            "defensive": 0.1,
            "offensive": 0.2
        }
        self.last_ball_position = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['last_ball_position'] = self.last_ball_position
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle.get('last_ball_position', 0)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward
        
        updated_rewards = reward.copy()
        components = {"base_score_reward": reward.copy(),
                      "midfield_dynamics": [0.0] * len(reward)}

        for i, o in enumerate(observation):
            # Check for repositioning
            ball_x_pos = o['ball'][0]
            movement = ball_x_pos - self.last_ball_position
            self.last_ball_position = ball_x_pos
            is_offensive = ball_x_pos > self.midfield_position and movement > 0
            is_defensive = ball_x_pos < self.midfield_position and movement < 0

            if is_offensive:
                updated_rewards[i] += self.repositioning_rewards["offensive"]
                components["midfield_dynamics"][i] += self.repositioning_rewards["offensive"]
            elif is_defensive:
                updated_rewards[i] += self.repositioning_rewards["defensive"]
                components["midfield_dynamics"][i] += self.repositioning_rewards["defensive"]

        return updated_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        info.update({f"component_{key}": sum(value) for key, value in components.items()})

        # Update sticky actions usage information
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_flag in enumerate(agent_obs['sticky_actions']):
                if action_flag:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
