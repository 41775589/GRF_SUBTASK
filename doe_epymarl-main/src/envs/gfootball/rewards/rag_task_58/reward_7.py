import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the defensive coordination response and transition from defense to attack."""

    def __init__(self, env):
        super().__init__(env)
        self.reset()
    
    def reset(self):
        """ Resets any necessary variables for the start of a new episode."""
        self.previous_ball_team_ownership = -1  # No team starts with the ball
        self.cumulative_defensive_rewards = np.zeros(10, dtype=np.float32)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['previous_ball_team_ownership'] = self.previous_ball_team_ownership
        return state

    def set_state(self, state):
        self.previous_ball_team_ownership = state['previous_ball_team_ownership']
        return self.env.set_state(state)

    def reward(self, reward):
        """ Modifies the reward function to account for defensive plays and counter attacks."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "defensive_play_reward": np.zeros_like(reward),
            "counter_attack_reward": np.zeros_like(reward)
        }

        for i, o in enumerate(observation):
            # Defensive play rewards: positive when successfully stealing or blocking the ball
            if o['ball_owned_team'] == 1 and self.previous_ball_team_ownership != 1:
                components["defensive_play_reward"][i] += 0.1
                self.cumulative_defensive_rewards[i] += 0.1
            
            # Counter attack rewards: when transitioning from defense (team 0) to possession (team 1)
            if self.previous_ball_team_ownership == 0 and o['ball_owned_team'] == 1:
                components["counter_attack_reward"][i] += 0.2
                self.cumulative_defensive_rewards[i] += 0.2

            reward[i] += components["defensive_play_reward"][i] + components["counter_attack_reward"][i]
        
        self.previous_ball_team_ownership = o['ball_owned_team']
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add the components to the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
