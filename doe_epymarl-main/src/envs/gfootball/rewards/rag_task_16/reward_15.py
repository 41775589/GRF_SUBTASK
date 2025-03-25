import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for precise high passes, focusing on technical skill enhancement.
    This wrapper will provide rewards based on the precision and appropriateness of high passes executed
    in scenarios where such a strategy is advantageous.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        components['high_pass_precision_reward'] = [0.0] * len(reward)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if high pass was performed
            if o['ball_direction'][2] > 0.1:  # Ball movement in z direction indicating a lift
                # Determine the accuracy of the pass
                ball_owner_team = o['ball_owned_team']
                ball_pos = o['ball']
                intended_target = o['left_team'] if ball_owner_team == 0 else o['right_team']
                
                # Calculate the distance and direction to all teammates to find the intended recipient
                distances = []
                for mate_pos in intended_target:
                    dist = np.linalg.norm(np.array(mate_pos)[:2] - np.array(ball_pos)[:2])
                    distances.append(dist)
                
                # Reward for shortest distance to any teammate, smaller distance better execution
                if distances:
                    min_distance = min(distances)
                    if min_distance < 0.3:  # Threshold for considering pass to be precise
                        components['high_pass_precision_reward'][rew_index] = 1.0 - min_distance

        final_rewards = [
            reward[i] + components['high_pass_precision_reward'][i]
            for i in range(len(reward))
        ]
        
        return final_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        sticky_actions_counter = self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
