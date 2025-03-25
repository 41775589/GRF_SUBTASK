import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that encourages agents to execute effective offensive maneuvers quickly 
    during varied phases of the game, focusing on positioning, quick attacks, and adaptability.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.quick_attack_bonus = 0.5
        self.dynamic_play_bonus = 0.3
        self.positioning_reward = 0.2
        self.game_mode_last_state = 0

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.game_mode_last_state = 0
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
                      "quick_attack_bonus": [0.0] * len(reward),
                      "dynamic_play_bonus": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            obs = observation[i]

            # Encourage quick progression towards goal when in possession
            if obs['ball_owned_team'] == 1:
                if obs['ball'][0] > 0.5:  # deep in opponent's half
                    components["quick_attack_bonus"][i] = self.quick_attack_bonus
                    reward[i] += components["quick_attack_bonus"][i]

            # Reward adaptability during game phase changes
            if obs['game_mode'] != self.game_mode_last_state:
                components["dynamic_play_bonus"][i] = self.dynamic_play_bonus
                reward[i] += components["dynamic_play_bonus"][i]

            # Extra reward for favorable positioning during normal play
            if obs['game_mode'] == 0:  # normal mode 
                player_position = obs['right_team'][obs['active']]
                # Boost for being in the right place at the right time
                if player_position[0] > 0:  # Player is on the opponent’s half
                    components["positioning_reward"][i] = self.positioning_reward
                    reward[i] += components["positioning_reward"][i]

        # Save the current game mode state for the next step
        self.game_mode_last_state = observation[0]['game_mode']
        
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
                self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
