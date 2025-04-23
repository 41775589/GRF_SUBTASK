import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dribbling and feinting skills improvement in a 1vs1 scenario with the goalkeeper."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.dribble_counter = np.zeros(2, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.feint_bonus = 0.05

    def reset(self):
        self.dribble_counter = np.zeros(2, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['dribble_counter'] = self.dribble_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribble_counter = np.array(from_pickle['dribble_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "feint_bonus": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]

            # Reward for dribbling and feinting when close to the goalkeeper.
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:  # Controlled by agent
                player_pos = o['left_team'][o['active']]
                goalkeeper_pos = o['right_team'][0]  # Assuming goalkeeper is always the first player

                distance_to_goalkeeper = np.linalg.norm(player_pos - goalkeeper_pos)

                # Close encounter dribbling bonus
                if distance_to_goalkeeper < 0.1:
                    dribble_action = o['sticky_actions'][9]  # index of dribble action in sticky actions
                    if dribble_action == 1:
                        self.dribble_counter[i] += 1
                        components["feint_bonus"][i] = self.dribble_counter[i] * self.feint_bonus
                        reward[i] += components["feint_bonus"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
