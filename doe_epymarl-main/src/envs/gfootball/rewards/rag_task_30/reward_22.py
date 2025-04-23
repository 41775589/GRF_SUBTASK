import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to add strategic positioning and movement pattern rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, rewards):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": rewards.copy(),
                      "defensive_positioning": [0.0] * len(rewards),
                      "counterattack_transition": [0.0] * len(rewards)}

        if observation is None:
            return rewards, components

        for i in range(len(rewards)):
            o = observation[i]
            # Implement defensive positioning reward: Encourage agents to maintain positions that block opponent advances
            if o['active'] != -1:  # Ensure we have a valid active player index
                player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
                opponent_goal = np.array([1, 0]) if o['ball_owned_team'] == 0 else np.array([-1, 0])
                opponent_players = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
                
                distance_to_goal = np.linalg.norm(player_pos - opponent_goal)
                distances_to_opponents = np.linalg.norm(opponent_players - player_pos, axis=1)
                min_distance_to_opponent = np.min(distances_to_opponents)

                # Reward for good defensive positioning (closer to goal and farther from opponents is better)
                components["defensive_positioning"][i] = 0.1 * (1 - distance_to_goal) + 0.1 * min_distance_to_opponent
            
            # Implement counterattack transition reward: Reward quick plays that reverse the play direction while in possession
            if o['ball_owned_team'] != -1 and self.sticky_actions_counter[8] == 1:  # Checks if 'sprint' action is active
                components["counterattack_transition"][i] = 0.5  # Reward for initiating a sprint while in possession

            # Update reward with the factors calculated
            rewards[i] += components["defensive_positioning"][i] + components["counterattack_transition"][i]

        return rewards, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter based on current actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        
        return obs, reward, done, info
