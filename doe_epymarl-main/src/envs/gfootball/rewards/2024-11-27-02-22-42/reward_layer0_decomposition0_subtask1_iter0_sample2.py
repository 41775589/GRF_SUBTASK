import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward specifically designed for defensive skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.distance_reward_factor = 0.1
        self.defensive_action_reward = 0.5

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
        base_score_reward = reward.copy()
        defensive_reward = [0.0] * len(reward)

        if observation is None:
            return reward, {'base_score_reward': base_score_reward, 'defensive_reward': defensive_reward}

        assert len(reward) == len(observation)  # Ensure alignment with original environment's structure

        for rew_index, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Consider only the defensive team
                # Reward for keeping the opponent far away from goal (distance based)
                opponent_positions = o['right_team']
                goal_position = [1, 0]  # Goal position on right side for left team
                min_distance = np.min([np.linalg.norm(player - goal_position) for player in opponent_positions])
                defensive_reward[rew_index] = self.distance_reward_factor / (min_distance + 0.1)
                
                # Check for specific defensive actions
                # Assuming 'sticky_actions' represents whether certain actions are being taken:
                # Index 0: Slide, Index 3: Sprint, Index 7: Stop Sprint, Index 1: Short Pass
                sticky_actions = o['sticky_actions']
                if sticky_actions[0] or sticky_actions[1] or sticky_actions[3] or sticky_actions[7]:
                    defensive_reward[rew_index] += self.defensive_action_reward

            # Update base reward considering additional rewards
            reward[rew_index] = base_score_reward[rew_index] + defensive_reward[rew_index]

        return reward, {'base_score_reward': base_score_reward, 'defensive_reward': defensive_reward}

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)
        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
