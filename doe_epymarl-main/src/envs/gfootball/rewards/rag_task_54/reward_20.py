import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a collaborative plays reward based on successful passes leading
    towards scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_to_shoot_ratio = 0.2  # Contribution of successful passes towards the goal-oriented plays
        self.shooting_threshold = 0.1   # Threshold for considering a position as a shooting opportunity

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle
        
    def reward(self, reward):
        """
        This method enhances the reward mechanism by considering how the ball is passed
        amongst players and how it leads to scoring opportunities.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_shoot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            
            # Check for scoring
            if reward[rew_index] == 1:  # Assuming reward of 1 for scoring a goal
                components["pass_shoot_reward"][rew_index] = 1.0
                reward[rew_index] += self.pass_to_shoot_ratio * components["pass_shoot_reward"][rew_index]
                continue
            
            # Calculating passing effectiveness
            if obs['ball_owned_team'] == 0: # The team of the agent is in possession of the ball
                ball_x, ball_y = obs['ball'][0], obs['ball'][1]
                for teammate in obs['left_team']:
                    # Distance check to see if players are in potential shooting range
                    if ball_x > (1 - self.shooting_threshold):
                        components["pass_shoot_reward"][rew_index] = 0.5
                        reward[rew_index] += self.pass_to_shoot_ratio * components["pass_shoot_reward"][rew_index]
            
        return reward, components
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Adding reward components to info for better monitoring
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action if action > 0 else 0
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
