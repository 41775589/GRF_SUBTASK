import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances teamwork and coordination in defensive strategies,
    focusing on maintaining ball control and executing strategic defensive positioning.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initializing custom tracking parameters for dynamic defensive positions
        self.team_defensive_configurations = {
            0: {'critical_positions': [], 'engaged_in_defense': False},
            1: {'critical_positions': [], 'engaged_in_defense': False}
        }
        self.defense_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.team_defensive_configurations
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.team_defensive_configurations = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Base score and initializing reward component dict.
        components = {"base_score_reward": reward.copy(),
                      "defense_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            own_team = observation[rew_index]['left_team'] if observation[rew_index]['ball_owned_team'] == 0 else observation[rew_index]['right_team']
            opposing_team = observation[rew_index]['right_team'] if observation[rew_index]['ball_owned_team'] == 0 else observation[rew_index]['left_team']

            if o['active'] < 0:
                continue

            # Engage in defense when opposing team holds the ball
            if o['ball_owned_team'] != o['ball_owned_team']:
                x_ball, y_ball = o['ball'][0], o['ball'][1]
                distances = [np.linalg.norm([x_ball - x, y_ball - y]) for x, y in opposing_team]
                min_index = np.argmin(distances)

                if distances[min_index] < 0.2:  # ball is very close to the player
                    self.team_defensive_configurations[o['ball_owned_team']]['critical_positions'].append(opposing_team[min_index])
                    self.team_defensive_configurations[o['ball_owned_team']]['engaged_in_defense'] = True
                    components['defense_reward'][rew_index] += self.defense_reward

            # Provide defense reward only when actively engaged in defense and the ball is lost to an opponent
            if self.team_defensive_configurations[o['ball_owned_team']]['engaged_in_defense'] and o['ball_owned_team'] != -1:
                reward[rew_index] += 1.5 * components['defense_reward'][rew_index]
                self.team_defensive_configurations[o['ball_owned_team']]['engaged_in_defense'] = False  # Reset defensive stance per episode

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
