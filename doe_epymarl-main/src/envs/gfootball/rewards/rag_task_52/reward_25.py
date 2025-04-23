import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to reinforce defensive skills including tackling, efficient movements,
    and pressured passing by providing step-by-step feedback through rewards.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.2
        self.efficient_movement_bonus = 0.05
        self.pressure_pass_bonus = 0.1

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
            "tackle_reward": [0.0] * len(reward),
            "movement_bonus": [0.0] * len(reward),
            "pressure_pass_bonus": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            ob = observation[rew_index]
            base_reward = reward[rew_index]
            sticky = ob['sticky_actions']

            # Reward efficient movement by checking if the player is moving less in sticky actions
            if not np.any(sticky[:8]) and ob['ball_owned_team'] == -1:  # Not moving, and not in possession of the ball
                components["movement_bonus"][rew_index] = self.efficient_movement_bonus
                reward[rew_index] += components["movement_bonus"][rew_index]

            # Reward for successful tackles (simulated by intercepting while the other team has ball control)
            if ob['ball_owned_team'] != 0 and last_action_was_tackle(sticky):  # Assuming function last_action_was_tackle denotes defensive actions
                components["tackle_reward"][rew_index] = self.tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]
            
            # Reward for pressured pass (assuming the player passed under a defensive context)
            if passed_under_pressure(ob, self.env.last_observation):  # Assuming hypothetical function to infer this
                components["pressure_pass_bonus"][rew_index] = self.pressure_pass_bonus
                reward[rew_index] += components["pressure_pass_bonus"][rew_index]

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

def last_action_was_tackle(sticky_actions):
    # Hypothetical check for tackle-like actions
    return sticky_actions[7] == 1  # Example index for "action_bottom_left" as a proxy for tackle

def passed_under_pressure(current_obs, prev_obs):
    # Hypothetical evaluation whether a pass was made under pressure
    ball_change = np.linalg.norm(current_obs['ball'] - prev_obs['ball'])
    opposing_players_close = np.min(np.linalg.norm(current_obs['left_team'] - current_obs['ball'], axis=1)) < 0.1
    return ball_change > 0.3 and opposing_players_close
