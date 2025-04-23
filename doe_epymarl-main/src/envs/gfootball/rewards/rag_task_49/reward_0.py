import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a shooting-focused reward from specific central field positions.
    The task emphasizes shooting with accuracy and power when aligned with the goal from midfield.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_zones = [(0.3, 0.6), (-0.6, -0.3)]  # Central zones from where shooting is optimal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shooting_zones_rewards'] = self.shooting_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shooting_zones = from_pickle['shooting_zones_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy_reward": [0.0, 0.0]}
        if observation is None:
            return reward, components

        for agent_index in range(len(reward)):
            obs = observation[agent_index]
            ball_pos = obs['ball'][:2]  # We consider x, y coordinates of the ball
            player_pos = obs['left_team'][obs['active']] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']]

            in_shooting_zone = any(z[0] < player_pos[0] < z[1] for z in self.shooting_zones)
            aligned_with_goal = abs(player_pos[1]) < 0.05  # ensure y coordinate is aligned with goalposts

            if in_shooting_zone and aligned_with_goal:
                # Increasing the shooting reward if in the optimal position to shoot
                components['shooting_accuracy_reward'][agent_index] += 0.5

                # Bump up reward when successfully shooting at the goal from these positions
                if obs['ball_direction'][0] > 0 and obs['ball_owned_team'] == 0:  # left team shoots right
                    reward[agent_index] += 1.0
                elif obs['ball_direction'][0] < 0 and obs['ball_owned_team'] == 1:  # right team shoots left
                    reward[agent_index] += 1.0

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
