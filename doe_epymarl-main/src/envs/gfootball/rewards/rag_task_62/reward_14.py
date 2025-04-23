import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a task-specific reward for shooting accuracy and timing near the goal."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize shooting zone threshold and accuracy importance
        self.shooting_zone_threshold = 0.2  # Close to the opponent's goal
        self.shooting_accuracy_weight = 5.0  # Emphasis on shooting accuracy
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball']
            distance_to_goal = np.abs(ball_pos[0] - 1)  # Assuming playing from left to right
            
            # Check if the ball is close to the opponent's goal and the action is a shot
            if distance_to_goal < self.shooting_zone_threshold and o['game_mode'] == 6:  # Game mode 6 is a penalty shot
                # Check if the shot led to a goal
                if reward[rew_index] > 0:
                    components["shooting_accuracy_reward"][rew_index] = self.shooting_accuracy_weight
                    reward[rew_index] += components["shooting_accuracy_reward"][rew_index]
                    
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
