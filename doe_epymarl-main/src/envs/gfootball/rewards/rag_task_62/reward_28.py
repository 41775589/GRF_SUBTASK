import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on shooting angles and timing near the goal."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = None  # No specific state to save here
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        # Initialize the additional reward components for each agent
        components["angle_timing_reward"] = [0.0] * len(reward)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o is None or 'ball' not in o or 'goal' not in o:
                continue
            
            ball_pos = np.array(o['ball'][:2])
            goal_pos = np.array([1, 0])  # Assuming goal at center of opponent's side
            player_pos = np.array(o['left_team'][o['active']])

            goal_vector = goal_pos - player_pos
            ball_vector = ball_pos - player_pos
            cos_angle = np.dot(goal_vector, ball_vector) / (np.linalg.norm(goal_vector) * np.linalg.norm(ball_vector))
            # Reward for aligning towards the goal with the ball
            angle_reward = max(0, cos_angle)
            
            # Timing reward: Closer to the goal gives higher reward during an attack mode
            if o['game_mode'] == 0:  # Normal game mode
                distance_to_goal = np.linalg.norm(goal_pos - player_pos)
                timing_reward = max(0, 1 - distance_to_goal)
            else:
                timing_reward = 0

            # Combining angle and timing rewards with a suitable weight to emphasize skillful shoot preparation
            components["angle_timing_reward"][rew_index] = 0.5 * angle_reward + 0.5 * timing_reward
            reward[rew_index] += components["angle_timing_reward"][rew_index]

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
            for i, active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += int(active)
        return observation, reward, done, info
