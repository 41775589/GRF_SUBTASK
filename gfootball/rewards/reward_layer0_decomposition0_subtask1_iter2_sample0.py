import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A modified reward wrapper to optimize defensive strategy training in soccer agents."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Adjusting the reward coefficients based on prior learning outcomes
        self.base_score_coefficient = 1.0
        self.defensive_action_coefficient = 0.2
        self.position_control_coefficient = 0.05

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
            'defensive_action_reward': [0.0] * len(reward),
            'position_control_reward': [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:
                actions = o['sticky_actions']
                # Simplified action rewards for defensive actions
                if any(actions[i] for i in [0, 1, 3, 7]):  # Assuming indices for certain actions
                    components['defensive_action_reward'][rew_index] = self.defensive_action_coefficient
                    
                # Strategy for positioning: rewarding distances to ball and opponent's goal
                goal_position = np.array([1, 0])  # Assuming this is the position of the opponent's goal
                player_position = o['left_team']
                ball_position = np.array(o['ball'][:2])

                # Reward for being between the ball and the goal
                for position in player_position:
                    dist_to_ball = np.linalg.norm(ball_position - position)
                    dist_to_goal = np.linalg.norm(goal_position - position)
                    control_score = (1 / (dist_to_ball + 1) + 1 / (dist_to_goal + 1))
                    components['position_control_reward'][rew_index] += self.position_control_coefficient * control_score

            # Final reward calculation with all components
            reward[rew_index] = (self.base_score_coefficient * components['base_score_reward'][rew_index] +
                                components['defensive_action_reward'][rew_index] +
                                components['position_control_reward'][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
