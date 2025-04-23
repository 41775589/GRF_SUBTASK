import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that promotes wingers to develop high-speed dribbling 
    and accurate crossing from the wings."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.wing_cross_coefficient = 3.0  # Reward multiplier for successful crosses
        self.sprint_coefficient = 1.2  # Reward multiplier for sprinting down wing
        self.cross_threshold = 0.5  # Threshold for considering a position as wing for crossing

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "cross_bonus": [0.0] * len(reward),
                      "sprint_bonus": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            base_reward = reward[rew_index]

            # Sprint reward: Check if the player is sprinting along the wings
            if o['sticky_actions'][8] == 1:  # action_sprint
                player_pos = o['right_team'][o['active']][:2]
                if np.abs(player_pos[1]) > self.cross_threshold:  # y position check for wings
                    components["sprint_bonus"][rew_index] = self.sprint_coefficient
                    base_reward += components["sprint_bonus"][rew_index]

            # Crossing reward: Reward successful crosses from the wings
            if o['game_mode'] == 4:  # e_GameMode_Corner
                players_pos = o['right_team'][:, 1]
                wing_players = np.where(np.abs(players_pos) > self.cross_threshold)[0]
                if o['ball_owned_player'] in wing_players:
                    components["cross_bonus"][rew_index] = self.wing_cross_coefficient
                    base_reward += components["cross_bonus"][rew_index]

            # Update the reward for this player
            reward[rew_index] = base_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Aggregate final reward and components data for debugging
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
