import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward function to focus on defensive skills with adjusted strategy."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.distance_reward_coefficient = 0.2
        self.action_reward_coefficient = 0.3

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            'base_score_reward': reward.copy(),
            'defensive_distance_reward': [0.0] * len(reward),
            'defensive_action_reward': [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if 'right_team' in o:  # Assume defense against 'right_team'
                # Enhance reward based on maintaining distance to the goal area by defense
                opponent_positions = o['right_team']
                goal_position = [1, 0]  # Approximations for goal position
                min_distance = np.min([np.linalg.norm(player - goal_position) for player in opponent_positions])
                components['defensive_distance_reward'][rew_index] = self.distance_reward_coefficient / (min_distance + 0.1)

                # Reward specific defensive actions
                sticky_actions = o.get('sticky_actions', [])
                if sticky_actions:
                    # Indices might be hypothetical; adjust based on actual environment detail indexing
                    if sticky_actions[0]:  # Suppose index 0 is 'sliding'
                        components['defensive_action_reward'][rew_index] += self.action_reward_coefficient
                    if sticky_actions[3]:  # Suppose index 3 is 'sprint'
                        components['defensive_action_reward'][rew_index] += self.action_reward_coefficient
                    if sticky_actions[7]:  # Suppose index 7 is 'stop sprint'
                        components['defensive_action_reward'][rew_index] += self.action_reward_coefficient

            # Combine all components to create final modified reward
            reward[rew_index] = (components['base_score_reward'][rew_index] +
                                 components['defensive_distance_reward'][rew_index] +
                                 components['defensive_action_reward'][rew_index])

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final adjusted reward to the info
        info["final_reward"] = sum(reward)
        
        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
