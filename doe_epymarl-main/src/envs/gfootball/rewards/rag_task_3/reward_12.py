import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for practicing shooting skills with accuracy and power."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the wrapper along with the environment's state."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the wrapper along with the environment's state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Enhance the reward function by focusing on shooting effectiveness."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            # Check if a goal is scored
            if o['score'][1] > o['score'][0]:  # Assuming the agent is on the right team
                reward[rew_index] += 100  # reward for scoring
            
            # Consider the ball control and shot power
            if o['ball_owned_team'] == 1:  # ball owned by the right team (agent's team)
                # Determine shooting situation based on proximity to the goal
                x_dist_to_goal_left = 1 - o['ball'][0]
                
                # Reward for shooting with power when close to the opponent's goal
                if x_dist_to_goal_left < 0.2:
                    reward[rew_index] += 5  # reward for shooting close to the goal
                
                # Bonus for controlled play and preparation before shooting
                if o['sticky_actions'][8] == 1 and o['sticky_actions'][9] == 1:  # sprint and dribble
                    reward[rew_index] += 1  # mild reward for dribbling with speed while preparing to shoot

        return reward, components

    def step(self, action):
        """Take a step in the environment and modify the returned reward."""
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
