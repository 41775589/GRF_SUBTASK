import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances gameplay strategy understanding and control, 
    especially for midfield roles, emphasizing positional play and teamwork."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.midfield_weight = 0.1
        self.position_bonus = [0.2, 0.1, 0.05]  # Central, wide, backline bonuses
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['MidfieldDynamics'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['MidfieldDynamics']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Calculate midfield control reward
            for player_idx in range(len(o['right_team'])):
                player_pos = o['right_team'][player_idx]  # Use right_team for example
                player_role = o['right_team_roles'][player_idx]
                
                # Check if the player is in midfield roles: CM (5 = Central Midfield)
                if player_role == 5:  
                    reward[rew_index] += self.midfield_weight

                    # Temporal distance bonus for being in significant midfield positions
                    x_pos = player_pos[0]
                    if abs(x_pos) < 0.2:  # Central
                        reward[rew_index] += self.position_bonus[0]
                    elif abs(x_pos) < 0.5:  # Wide areas
                        reward[rew_index] += self.position_bonus[1]
                    else:  # Back towards own half
                        reward[rew_index] += self.position_bonus[2]

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
