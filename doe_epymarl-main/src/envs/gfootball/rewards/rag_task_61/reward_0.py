import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for strategic positioning and possession change recognition."""
    
    def __init__(self, env):
        super().__init__(env)
        self.goal_possession_change = 0.5
        self.bonus_per_player_position = 0.1
        self.possession_last_step = None
        self.player_positions_last_step = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_last_step = None
        self.player_positions_last_step = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['possession_last_step'] = self.possession_last_step
        to_pickle['player_positions_last_step'] = self.player_positions_last_step
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.possession_last_step = from_pickle['possession_last_step']
        self.player_positions_last_step = from_pickle['player_positions_last_step']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Process rewards based on strategic positioning and possession change
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['ball_owned_team'] != self.possession_last_step:
                # Reward for changing possession
                if self.possession_last_step is not None:
                    components["possession_change_reward"][rew_index] += self.goal_possession_change
                    reward[rew_index] += self.goal_possession_change
            
            # Update possession status for the next step
            self.possession_last_step = o['ball_owned_team']

            # Reward based on player positioning during the positional play
            num_players_in_position = sum(
                [1 for i, pos in enumerate(o['left_team'] if o['left_team_active'][i]) if pos[0] > 0.5] +
                [1 for i, pos in enumerate(o['right_team'] if o['right_team_active'][i]) if pos[0] < -0.5]
            )
            components["positioning_reward"][rew_index] = num_players_in_position * self.bonus_per_player_position
            reward[rew_index] += components["positioning_reward"][rew_index]

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
