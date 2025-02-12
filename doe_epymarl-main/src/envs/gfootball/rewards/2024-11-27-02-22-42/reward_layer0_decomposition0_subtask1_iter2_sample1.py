import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper for optimizing defensive skills in a football environment."""

    def __init__(self, env):
        super().__init__(env)
        # Define coefficients for each reward component
        self.position_coefficient = 0.2
        self.action_coefficient = 0.2
        self.base_score_coefficient = 1.0
        self.distance_from_goal_weight = 0.1
        self.defensive_success_weight = 0.5

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
        action_reward = [0.0] * len(reward)
        position_reward = [0.0] * len(reward)
        distance_from_goal_reward = [0.0] * len(reward)
        defensive_success_reward = [0.0] * len(reward)

        if observation is None:
            return reward, {'base_score_reward': base_score_reward}

        for rew_index, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0:
                # Increased reward for defensive actions
                if obs['sticky_actions'][0] or obs['sticky_actions'][3]:  # Consider tackling or sprint actions
                    action_reward[rew_index] = self.action_coefficient

                # Incentivize positioning between the ball and the goal
                player_position = obs['left_team']
                ball_position = np.array(obs['ball'][:2])
                goal_position = np.array([1, 0])  # Assuming goal x-location at 1 and centered y
                player_to_goal_dist = np.linalg.norm(player_position - goal_position)
                ball_to_goal_dist = np.linalg.norm(ball_position - goal_position)

                if player_to_goal_dist < ball_to_goal_dist:
                    position_reward[rew_index] += self.position_coefficient

                distance_from_goal_reward[rew_index] = -self.distance_from_goal_weight * player_to_goal_dist

                # Defensive success: prevent the opponent from scoring
                if obs['score'][0] == 0:
                    defensive_success_reward[rew_index] = self.defensive_success_weight

            final_component_rewards = (
                self.base_score_coefficient * base_score_reward[rew_index] +
                action_reward[rew_index] +
                position_reward[rew_index] +
                distance_from_goal_reward[rew_index] +
                defensive_success_reward[rew_index]
            )
            reward[rew_index] = final_component_rewards

        return reward, {
            'base_score_reward': base_score_reward,
            'action_reward': action_reward,
            'position_reward': position_reward,
            'distance_from_goal_reward': distance_from_goal_reward,
            'defensive_success_reward': defensive_success_reward
        }

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
