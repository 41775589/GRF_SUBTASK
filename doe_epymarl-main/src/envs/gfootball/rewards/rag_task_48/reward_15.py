import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards high passes from midfield which lead to direct scoring opportunities.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_threshold = 0.2  # Defines the midfield region boundary
        self.high_pass_reward = 1.0  # Reward for a successful high pass
        self.scoring_opportunity_multiplier = 2.0  # Bonus if the high pass creates a scoring chance

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        # Initialize component arrays
        components["high_pass_reward"] = [0.0, 0.0]
        components["scoring_opportunity_reward"] = [0.0, 0.0]

        for i in range(len(reward)):
            o = observation[i]
            ball_pos = o['ball'][0]  # Get the x-coordinate of the ball

            # Check if the ball is in the new position within the midfield area
            if abs(ball_pos) < self.midfield_threshold:
                
                if o['game_mode'] == 4:  # Check if the last action was a high pass (corner game mode as an example)
                    components["high_pass_reward"][i] = self.high_pass_reward
                    reward[i] += components["high_pass_reward"][i]
                    
                    # Check if this pass leads to a direct scoring opportunity
                    if o['score'][0] > o['score'][1]:  # Simple method to check if a goal was scored
                        components["scoring_opportunity_reward"][i] = self.high_pass_reward * self.scoring_opportunity_multiplier
                        reward[i] += components["scoring_opportunity_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
