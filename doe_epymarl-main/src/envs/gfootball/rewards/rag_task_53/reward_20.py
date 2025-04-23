import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """This reward wrapper focuses on promoting ball control under pressure, strategic play, and effective distribution
    across the football field."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Defining scalability constants for reward components.
        self.ball_control_under_pressure_reward = 0.1
        self.strategic_play_reward = 0.05
        self.effective_pass_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_control_under_pressure": [0.0] * len(reward),
                      "strategic_play": [0.0] * len(reward),
                      "effective_pass": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for maintaining ball control under pressure
            if o['ball_owned_team'] == o['active'] and o['left_team_active'].sum() < 2:
                components["ball_control_under_pressure"][rew_index] = self.ball_control_under_pressure_reward
            
            # Reward for strategic play: making space or moving the ball near opponents' box
            if np.any(o['right_team'][:, 0] > 0.9):  # any opponent player past x=0.9
                components["strategic_play"][rew_index] = self.strategic_play_reward
            
            # Reward for effective distribution by observing passes leading to significant advancement
            if o['ball_direction'][0] > 0.05:  # significant forward movement along x-axis
                components["effective_pass"][rew_index] = self.effective_pass_reward

            # Combine all these specifics into a final modified reward for this agent
            reward[rew_index] += (components["ball_control_under_pressure"][rew_index] +
                                  components["strategic_play"][rew_index] +
                                  components["effective_pass"][rew_index])

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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
