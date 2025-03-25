import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds advanced rewards based on energy conservation strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.stamina_threshold = 0.8  # Stamina must be above this threshold
        self.sprint_penalty = -0.1    # Penalty for sprinting below threshold
        self.stop_moving_bonus = 0.2  # Bonus for stopping when advisable
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        components["stamina_rewards"] = [0.0] * len(reward)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            stamina = o['right_team_tired_factor'][o['active']] if o['ball_owned_team'] == 1 else o['left_team_tired_factor'][o['active']]

            # Penalize sprint usage when stamina is low
            if o['sticky_actions'][8] == 1 and stamina < self.stamina_threshold:
                components["stamina_rewards"][rew_index] += self.sprint_penalty
                reward[rew_index] += components["stamina_rewards"][rew_index]

            # Reward stopping movement to conserve energy when not necessary to move
            if np.array_equal(o['right_team_direction'][o['active']], [0, 0]) or np.array_equal(o['left_team_direction'][o['active']], [0, 0]):
                components["stamina_rewards"][rew_index] += self.stop_moving_bonus
                reward[rew_index] += components["stamina_rewards"][rew_index]

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
