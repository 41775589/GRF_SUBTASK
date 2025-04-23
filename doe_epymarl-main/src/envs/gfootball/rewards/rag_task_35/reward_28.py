import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards in football based on maintaining strategic positions and effectively transitioning between defensive stance and attacking movements."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        components = {'base_score_reward': reward.copy()}
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Adjust rewards for maintaining strategic positions and transitioning 
            strategic_coefficient = 0.01
            attack_position_factor = 0.05
            defense_position_factor = 0.05

            # Estimate positioning benefit (focus on x-axis strategic location)
            middle_field = 0     # Neutral field position
            goal_attack = 1      # Close to opponent goal
            goal_defend = -1     # Close to home goal

            player_x_positions = o['right_team'][:, 0] if o['ball_owned_team'] == 1 else o['left_team'][:, 0]

            # Encourage moving forward from a defensive position towards middle and attacking zones
            if np.any(player_x_positions < middle_field):
                defense_positioning_reward = defense_position_factor * np.sum(goal_attack - player_x_positions[player_x_positions < middle_field])
            else:
                defense_positioning_reward = 0

            # Encourage strategic attacking positions
            if np.any(player_x_positions > middle_field):
                attack_positioning_reward = attack_position_factor * np.sum(player_x_positions[player_x_positions > middle_field] - middle_field)
            else:
                attack_positioning_reward = 0

            strategic_reward = strategic_coefficient * (defense_positioning_reward + attack_positioning_reward)
            components.setdefault('strategic_position_reward', []).append(strategic_reward)

            # Update the reward for this player
            reward[rew_index] += strategic_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Include sticky actions visibility in the info dictionary
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, act_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += act_state
                info[f"sticky_actions_{i}"] = act_state

        return observation, reward, done, info
