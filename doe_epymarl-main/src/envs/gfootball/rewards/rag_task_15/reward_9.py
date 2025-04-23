import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper emphasizing the practice and mastering of long passes,
    calculating rewards based on ball travel distance and accuracy.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_start_pos = None
        self.last_ball_owner_team = -1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_start_pos = None
        self.last_ball_owner_team = -1
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_accuracy_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if ('ball_owned_team' in o) and (o['ball_owned_team'] in [0, 1]):
                # Check if there is a change in ball ownership indicating a pass
                if self.last_ball_owner_team != -1 and self.last_ball_owner_team != o['ball_owned_team']:
                    pass_end_pos = o['ball'][:2]  # Get x, y position of the ball
                    if self.pass_start_pos is not None:
                        # Calculate distance of the pass
                        distance = np.sqrt((pass_end_pos[0] - self.pass_start_pos[0])**2 + (pass_end_pos[1] - self.pass_start_pos[1])**2)
                        # Reward based on distance, longer passes are more rewarded
                        components['pass_accuracy_reward'][rew_index] = distance * 0.5
                        reward[rew_index] += components['pass_accuracy_reward'][rew_index]

                self.pass_start_pos = o['ball'][:2]  # Update the start position of the new pass
                self.last_ball_owner_team = o['ball_owned_team']

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
