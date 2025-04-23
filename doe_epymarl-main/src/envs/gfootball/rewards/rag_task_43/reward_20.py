import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on enhancing defensive strategies and counterattacks for football agents."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward coefficients
        self.defensive_position_coefficient = 1.0
        self.counterattack_speed_coefficient = 0.5
        self.ball_interception_coefficient = 1.5
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
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
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "counterattack_reward": [0.0] * len(reward),
            "interception_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for index, obs in enumerate(observation):
            # Calculate rewards for defensive positioning
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0:
                # Reward agents for being closer to their goal when defending
                own_goal_y = -1 if obs['active'] in obs['left_team'] else 1
                player_position = obs['left_team'][obs['active']] if obs['active'] in obs['left_team'] else obs['right_team'][obs['active']]
                distance_to_goal = abs(player_position[0] - own_goal_y)
                components["defensive_reward"][index] = self.defensive_position_coefficient * (1 - distance_to_goal)

            # Counterattack speed reward
            if 'ball_direction' in obs and 'ball_owned_team' in obs and obs['ball_owned_team'] == 1:
                speed_of_attack = np.linalg.norm(obs['ball_direction'][:2])
                components["counterattack_reward"][index] = self.counterattack_speed_coefficient * speed_of_attack

            # Ball interception reward
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == -1:
                # Check proximity to the ball for potential interception
                ball_position = obs['ball'][:2]
                player_position = obs['left_team'][obs['active']] if obs['active'] in obs['left_team'] else obs['right_team'][obs['active']]
                distance_to_ball = np.linalg.norm(player_position - ball_position)
                if distance_to_ball < 0.05:  # Threshold for 'close to intercepting'
                    components["interception_reward"][index] = self.ball_interception_coefficient

            # Aggregate the reward modifications
            reward[index] += (
                components["defensive_reward"][index] +
                components["counterattack_reward"][index] +
                components["interception_reward"][index]
            )

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
