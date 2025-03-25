import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = [(-1, 0.25), (-1, -0.25), (-0.75, 0), (-0.75, 0.25), (-0.75, -0.25)]
        self.position_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards.clear()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['position_rewards'] = self.position_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_rewards = from_pickle.get('position_rewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'positional_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Only rewarding the defensive strategy of the left team
            if o['active'] < 0 or o['ball_owned_team'] != 0:
                continue

            player_pos = o['left_team'][o['active']]

            # Loop through designated defensive zones and check if player is within any.
            for idx, def_pos in enumerate(self.defensive_positions):
                key = f'player_{rew_index}_pos_{idx}'
                if np.linalg.norm(np.array(def_pos) - player_pos[:2]) <= 0.1:
                    if key not in self.position_rewards:
                        self.position_rewards[key] = 1
                        components['positional_reward'][rew_index] += 0.05
                        reward[rew_index] += components['positional_reward'][rew_index]

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
