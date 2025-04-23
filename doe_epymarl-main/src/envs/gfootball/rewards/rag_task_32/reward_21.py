import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.checkpoint_positions = np.linspace(-1, 1, 11)  # Represents positions along the x-axis from left to right
        self.checkpoints_collected = {i: False for i in range(len(self.checkpoint_positions))}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.checkpoints_collected = {i: False for i in self.checkpoints_collected}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), 
                      "crossing_reward": 0, 
                      "sprint_reward": 0}
        
        observations = self.env.unwrapped.observation()
        for obs in observations:
            current_pos_x = obs['ball'][0]
            current_pos_y = obs['ball'][1]
            dribbling = obs['sticky_actions'][9]
            
            # Reward for crossing in the last third of the pitch near the goal area
            if 0.6 <= abs(current_pos_x) <= 1 and abs(current_pos_y) > 0.2:
                if not self.checkpoints_collected[int((current_pos_x + 1) * 5)]:
                    self.checkpoints_collected[int((current_pos_x + 1) * 5)] = True
                    components['crossing_reward'] += 1  # Positive reward for a cross in the promising area
            
            # Reward for sustained sprinting (dribbling with high speed)
            if dribbling and 'sprint' in obs['action'][8]:
                components['sprint_reward'] += 0.1

        reward += components['crossing_reward']
        reward += components['sprint_reward']
        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info.update({
            "final_reward": reward,
            "component_crossing_reward": components['crossing_reward'],
            "component_sprint_reward": components['sprint_reward']
        })

        obs = self.env.unwrapped.observation()  # Update environment observations after step
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_active
        return obs, reward, done, info
