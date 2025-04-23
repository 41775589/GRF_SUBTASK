import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on advanced ball control and passing under pressure."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Reward configuration
        self.pass_success_reward = 0.5
        self.control_under_pressure_reward = 0.3
        self.pass_attempt_penalty = -0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_success_reward": [0.0] * len(reward),
                      "control_under_pressure_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            # Calculate rewards based on ball control under pressure
            if obs['ball_owned_team'] == obs['active'] and obs['ball_owned_player'] >= 0:
                # Check if there are multiple opponents nearby
                own_position = obs['right_team'][obs['ball_owned_player']] if obs['active'] == 1 else obs['left_team'][obs['ball_owned_player']]
                opponent_positions = obs['left_team'] if obs['active'] == 1 else obs['right_team']
                opponents_nearby = np.sum(np.linalg.norm(opponent_positions - own_position, axis=1) < 0.1)

                if opponents_nearby >= 2:  # indicates high pressure
                    components["control_under_pressure_reward"][index] = self.control_under_pressure_reward
                    reward[index] += components["control_under_pressure_reward"][index]

            # Rewards for successful passing under pressure
            if 'last_action_success' in obs and obs['last_action_success']:
                pass_actions = obs['sticky_actions'][7:10]  # indices for pass actions: Short, High, Long pass
                if np.any(pass_actions):
                    components["pass_success_reward"][index] = self.pass_success_reward
                    reward[index] += components["pass_success_reward"][index]
            elif 'last_action_type' in obs and any(obs['sticky_actions'][7:10]):
                reward[index] += self.pass_attempt_penalty  # Penalty for unsuccessful pass

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info["component_" + key] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
