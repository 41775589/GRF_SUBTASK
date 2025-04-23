import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that emphasizes scoring and productive passes in strategic, game-like contexts."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.1
        self.goal_reward = 1.0
        
    def reset(self):
        """ Resets the environment and sticky counters. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """ Get the state of the environment for serialization """
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """ Set the state of the environment from deserialization """
        return self.env.set_state(state)

    def reward(self, reward):
        """ Calculate the augmented reward based on ball possession and successful passes towards the goal."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "goal_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check if the team scored a goal
            if o['score'][1] > o['score'][0]:  # Assuming 'score' is [team_0, team_1] and team_1 is the agent's team
                components["goal_reward"][rew_index] = self.goal_reward
                reward[rew_index] += components["goal_reward"][rew_index]

            # Check if a strategic pass was made
            pass_condition = (o['sticky_actions'][9] == 1 and  # Assuming 'sticky_actions' index 9 is for a pass
                              o['ball_owned_team'] == 1  # Assuming the ball is owned by the agent's team
                             )
            if pass_condition:
                potential_pass_effectiveness = self.analyze_pass_efficiency(o)
                components["pass_reward"][rew_index] = self.pass_reward * potential_pass_effectiveness
                reward[rew_index] += components["pass_reward"][rew_index]

        return reward, components
    
    def analyze_pass_efficiency(self, observation):
        """ Analyze the direction and impact of pass; currently simplistic logic. """
        ball_direction = observation['ball_direction'][:2]
        goal_direction = np.array([1, 0])  # Assuming the goal direction is always to the right
        efficiency = np.dot(ball_direction, goal_direction)  # Dot product indicating alignment with goal
        return max(0, efficiency)  # Ensuring no negative efficiency
    
    def step(self, action):
        """ Performs a step in the environment, adjusting the reward using custom wrapper logic. """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky action tracking
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = action_active
                    
        return observation, reward, done, info
