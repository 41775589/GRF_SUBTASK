import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward based on strategic long-range passing."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for "long" passes; a rough estimate might be half the field width.
        self.long_pass_threshold = 0.5
        self.pass_reward = 0.1
        
    def reset(self):
        # Reset the sticky action counter on new episodes.
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'passing_reward': [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            # Calculate the distance of the ball pass based on ball's previous and current position
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # only consider our team
                prev_ball_position = o.get('prev_ball_position', o['ball'])
                ball_movement = np.linalg.norm(o['ball'][:2] - prev_ball_position[:2])
                
                # Reward long passes effectively for strategic plays
                if ball_movement > self.long_pass_threshold and o['sticky_actions'][9]:
                    # Check if the ball movement was primarily horizontal, fitting a "strategic pass"
                    angle_of_pass = abs(o['ball'][1] - prev_ball_position[1]) / ball_movement
                    if angle_of_pass < 0.5:  # assuming 0.5 to filter out predominantly vertical passes
                        components['passing_reward'][rew_index] = self.pass_reward
                        reward[rew_index] += self.pass_reward

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
            if 'sticky_actions' in agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
