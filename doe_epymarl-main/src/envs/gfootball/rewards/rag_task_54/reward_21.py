import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances rewards for collaborative plays between shooters (players near the opponent goal)
    and passers (players making key passes). The idea is to encourage passing that leads to scoring opportunities.
    """
    def __init__(self, env):
        super().__init__(env)
        self.pass_threshold = 0.3  # distance threshold to consider a pass
        self.goal_zone_threshold = 0.7  # x-coordinate beyond which a player is considered near the goal
        self.rewards_for_pass = 0.1
        self.rewards_for_shooting_position = 0.2
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_threshold'] = self.pass_threshold
        to_pickle['goal_zone_threshold'] = self.goal_zone_threshold
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_threshold = from_pickle.get('pass_threshold', self.pass_threshold)
        self.goal_zone_threshold = from_pickle.get('goal_zone_threshold', self.goal_zone_threshold)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "pass_rewards": [0.0] * len(reward),
            "shooting_position_rewards": [0.0] * len(reward)
        }

        current_ball_owner_team = observation['ball_owned_team']
        current_ball_owner_player = observation['ball_owned_player']

        # Check for collaborative play rewards: passing and being in a shooting position
        for agent_index, agent_obs in enumerate(observation):
            agent_x_pos = agent_obs['position'][0]

            # Reward players in shooting positions near the opponent's goal
            if agent_x_pos > self.goal_zone_threshold:
                components["shooting_position_rewards"][agent_index] = self.rewards_for_shooting_position
                reward[agent_index] += components["shooting_position_rewards"][agent_index]

            # Check passes from one player to another leading to a potential shot
            if self.previous_ball_owner is not None and self.previous_ball_owner != current_ball_owner_player:
                pass_distance = np.linalg.norm(
                    observation[self.previous_ball_owner]['position'] -
                    agent_obs['position']
                )
                
                # If a pass is made to a player near the goal
                if pass_distance < self.pass_threshold and agent_x_pos > self.goal_zone_threshold:
                    components["pass_rewards"][self.previous_ball_owner] = self.rewards_for_pass
                    reward[self.previous_ball_owner] += components["pass_rewards"][self.previous_ball_owner]

        self.previous_ball_owner = current_ball_owner_player if current_ball_owner_team == observation['active_team'] else None

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
