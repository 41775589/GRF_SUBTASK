import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on offensive actions and positioning relative to the opponent's goal. 
    The reward promotes actions such as Shots, Dribbles, and successful Passes near the opponent's goal.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.num_checkpoints = 5
        self.checkpoint_reward = 0.2
        self.goal_y_positions = [-0.42, -0.21, 0, 0.21, 0.42]

    def reset(self, **kwargs):
        self.checkpoints_collected = [0]*self.num_agents
        return self.env.reset(**kwargs)

    def reward(self, reward):
        base_reward = reward.copy()
        components = {
            "base_score_reward": base_reward,
            "offensive_action_reward": [0.0] * len(reward),
            "goal_distance_reward": [0.0] * len(reward)
        }

        observations = self.env.unwrapped._get_obs()  # pretend method to retrieve observations directly
        for i in range(len(reward)):
            obs = observations[i]

            # Calculate the reward contribution for being close to opponent's goal with the ball
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                ball_pos_y = obs['ball'][1]
                closest_goal_y = min(self.goal_y_positions, key=lambda y: abs(y - ball_pos_y))
                distance_to_goal = abs(ball_pos_y - closest_goal_y)
                goal_index = self.goal_y_positions.index(closest_goal_y)
                
                if self.checkpoints_collected[i] <= goal_index:
                    components["goal_distance_reward"][i] += self.checkpoint_reward * (goal_index - self.checkpoints_collected[i])
                    self.checkpoints_collected[i] = goal_index + 1

            # Simple reward increments for desired offensive actions
            if obs['sticky_actions'][1]:  # Assuming index 1 corresponds to 'Shot'
                components["offensive_action_reward"][i] += 0.5
            if obs['sticky_actions'][2]:  # Assuming index 2 corresponds to 'Dribble'
                components["offensive_action_reward"][i] += 0.3
            if obs['sticky_actions'][3]:  # Assuming index 3 corresponds to 'Short Pass'
                components["offensive_action_reward"][i] += 0.1
            if obs['sticky_actions'][4]:  # Assuming index 4 corresponds to 'Long Pass'
                components["offensive_action_reward"][i] += 0.1

            # Combine all components of rewards
            reward[i] = sum(components[k][i] for k in components)

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)
        info.update({f"component_{key}": sum(value) for key, value in components.items()})
        info['final_reward'] = sum(modified_reward)
        return obs, modified_reward, done, info
