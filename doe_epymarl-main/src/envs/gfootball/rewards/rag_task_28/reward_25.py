import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds a reward for dribbling skills against the goalkeeper. """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.array([0., 0., 0.])
        self.dribbling_reward = 0.05  # Dribbling reward coefficient
        self.goal_distance_threshold = 0.2  # Proximity to goal to consider dribbling against the goalkeeper
        self.ball_control_reward_mul = 5  # Multiplier for ball control under pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.array([0., 0., 0.])
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['dribbling_state'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'last_ball_position': self.last_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'dribbling_state' in from_pickle:
            self.sticky_actions_counter = from_pickle['dribbling_state']['sticky_actions_counter']
            self.last_ball_position = from_pickle['dribbling_state']['last_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        new_rewards = [r for r in reward]  # Copy base reward list 
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward)
        }

        for i in range(len(reward)):
            o = observation[i]
            if o['ball_owned_team'] == 1:  # If the right team controls the ball
                ball_pos = o['ball'][:2]  # Get the current ball position, exclude z-axis
                proximity_to_goal = np.abs(ball_pos[0] - 1)
                has_ball_control = o['ball_owned_player'] == o['active']
                
                # Check if player is dribbling facing the goalkeeper within a threshold
                if proximity_to_goal <= self.goal_distance_threshold and has_ball_control:
                    # Calculate reward based on control and movement complexity
                    distance_moved = np.linalg.norm(ball_pos - self.last_ball_position[:2])
                    components['dribbling_reward'][i] = self.dribbling_reward * distance_moved * self.ball_control_reward_mul
                    new_rewards[i] += components['dribbling_reward'][i]

            self.last_ball_position = o['ball']  # Update last known ball position

        return new_rewards, components

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
