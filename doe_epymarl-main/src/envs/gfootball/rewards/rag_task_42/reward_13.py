import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances learning by adding rewards for midfield dynamics mastery, 
    including coordination under pressure and strategic positioning during offense and 
    defense transitions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.midfield_checkpoints = np.linspace(-0.3, 0.3, 10)
        self.coordination_bonus = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_coordination_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            if player_obs['active'] == -1:
                continue
            
            # Positional reward based on midfield control
            player_y_position = player_obs['left_team'][player_obs['active']][1]
            ball_y_position = player_obs['ball'][1]

            # Bonus for maintaining position in tough midfield regions under opponent pressure
            if abs(player_y_position) in self.midfield_checkpoints:
                components['midfield_coordination_reward'][rew_index] = self.coordination_bonus
                reward[rew_index] += components['midfield_coordination_reward'][rew_index]

            # Additional coordination reward if maintaining formation and ball possession under pressure
            if (player_obs['ball_owned_team'] == 0) and (player_obs['left_team_active'][player_obs['active']]):
                player_dir_magnitude = np.linalg.norm(player_obs['left_team_direction'][player_obs['active']])
                if player_dir_magnitude < 0.1:
                    reward[rew_index] += 0.1  # Reward for resisting pressure and holding the line

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
