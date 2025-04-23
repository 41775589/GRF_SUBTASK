import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward tailored for defending strategies."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_reward = 0.3
        self._passing_reward = 0.2
        self._movement_control_reward = 0.1

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
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward),
                      "movement_control_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index] if observation is not None else None

            if o is None:
                continue

            if o['game_mode'] in (0, 3):  # Normal play or Free Kick
                # Check possession and defensive actions
                if o['ball_owned_team'] == 1 and o['active'] in o['right_team_roles'] and o['sticky_actions'][0] > 0:
                    components['tackle_reward'][rew_index] = self._tackle_reward
                    reward[rew_index] += components['tackle_reward'][rew_index]

                # Efficient movement control (stopping techniques)
                if np.linalg.norm(o['right_team_direction'][o['active']]) <= 0.01:
                    components['movement_control_reward'][rew_index] = self._movement_control_reward
                    reward[rew_index] += components['movement_control_reward'][rew_index]

                # Assisting a pressured pass success
                if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == 0:
                    components['passing_reward'][rew_index] = self._passing_reward
                    reward[rew_index] += components['passing_reward'][rew_index]

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
