import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on advanced dribbling techniques. It delivers incremental
    rewards when an agent uses sprint actions effectively to dribble past defenders or move towards
    the goal without losing possession, portraying typical tight defense scenarios.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reward_for_sprint_usage = 0.1
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Modifies rewards based on the agent's effective use of sprint action while controlling 
        the ball and making progressive movements towards the opponent's goal.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_usage_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and 'active' in o:
                if o['sticky_actions'][9] == 1:  # Checking if sprint action is active
                    # Reward is given if the player is moving towards the opponent's goal
                    if o['ball'][0] > 0.5:  # Assuming ball's x > 0.5 is towards the opponent's goal
                        components["sprint_usage_reward"][rew_index] = self.reward_for_sprint_usage
                        reward[rew_index] += components["sprint_usage_reward"][rew_index]

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
