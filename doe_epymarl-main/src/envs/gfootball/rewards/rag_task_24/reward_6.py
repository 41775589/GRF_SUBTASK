import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for effective mid to long-range passing in a strategic and coordinated manner.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_success_reward = 0.3  # Reward for successful long pass
        self.strategy_bonus = 0.5  # Additional reward for strategic passes leading to goal opportunities
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Augments the vanilla reward with additional incentives for high-quality passing and strategic gameplay.
        
        reward : list of base rewards corresponding to the outcome of actions taken.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_success_reward": [0.0] * len(reward),
                      "strategy_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            current_obs = observation[rew_index]
            # Check high and long pass activity
            if 'ball_direction' in current_obs:
                # Vector norm for the "long pass" 
                ball_speed = np.linalg.norm(current_obs['ball_direction'][:2])
                
                # Condition for a long pass: significant x (long) and reasonable y (elevation) velocity
                if ball_speed > 0.1 and abs(current_obs['ball_direction'][2]) > 0.02:
                    components["pass_success_reward"][rew_index] = self.pass_success_reward
                    reward[rew_index] += components["pass_success_reward"][rew_index]
                    
                    # Check strategic play, e.g., leading to goal-scoring opportunities after such a pass
                    if 'ball_owned_team' in current_obs and current_obs['ball_owned_team'] == 1:  # Assuming right team is ours
                        opponent_goal_distance_after_pass = abs(1 - current_obs['ball'][0])
                        # Closer to goal and pass was initiated closer to the midfield or own field
                        if opponent_goal_distance_after_pass < 0.5 and current_obs['ball'][0] < 0:
                            components["strategy_bonus"][rew_index] = self.strategy_bonus
                            reward[rew_index] += components["strategy_bonus"][rew_index]
        
        return reward, components

    def step(self, action):
        """
        Step function to execute the environment's step and apply the custom reward adjustments.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key in components:
            info[f"component_{key}"] = sum(components[key])
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
