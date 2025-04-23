import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds checkpoint rewards based on defensive positioning and counterattack optimization."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define a step checkpoint for tracking whether the defenders are effectively stopping direct attacks.
        # This checkpoint encouraged positioning between the ball and the goal in defensive scenarios.
        self.defensive_positions_counter = 0
        self.counterattack_rewards = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions_counter = 0
        self.counterattack_rewards = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "defensive_positions": self.defensive_positions_counter,
            "counterattack_rewards": self.counterattack_rewards
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions_counter = from_pickle.get('CheckpointRewardWrapper', {}).get("defensive_positions", 0)
        self.counterattack_rewards = from_pickle.get('CheckpointRewardWrapper', {}).get("counterattack_rewards", 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_position_reward": 0.0,
            "counter_attack_reward": 0.0
        }

        if observation is None:
            return reward, components

        current_ball_position = np.array(observation['ball'][:2])
        left_team_positions = observation['left_team']
        left_goal = np.array([-1, 0])  # considering left goal position

        # Defensive positioning check: encouraging players to stay between the ball and their goal.
        defensive_positioning = np.all(
            np.linalg.norm(left_team_positions - left_goal, axis=1) < np.linalg.norm(current_ball_position - left_goal)
        )
        if defensive_positioning:
            self.defensive_positions_counter += 1
            reward += 0.1
            components["defensive_position_reward"] = 0.1

        # Counterattack reward: if the ball is taken over and quickly transitioned to an offensive opportunity.
        if observation['ball_owned_team'] == 0 and np.any(observation['right_team_direction'][:, 0] > 0):
            self.counterattack_rewards += 1
            reward += 0.2
            components["counter_attack_reward"] = 0.2

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += active
        return observation, reward, done, info
