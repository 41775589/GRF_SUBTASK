import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances defensive unit capabilities and encourages strategic positioning.
    The reward emphasizes maintaining defensive formations and effective counterattacks against direct attacks.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions_achieved = 0
        self.ball_interceptions = 0
        self.confrontational_def_reward = 0.2
        self.positional_play_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions_achieved = 0
        self.ball_interceptions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_positions'] = self.defensive_positions_achieved
        to_pickle['ball_interceptions'] = self.ball_interceptions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions_achieved = from_pickle['defensive_positions']
        self.ball_interceptions = from_pickle['ball_interceptions']
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": 0.0, "positional_play_reward": 0.0}
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components

        for o in observation:
            # Receiving ball represents interception in defensive scenarios
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                self.ball_interceptions += 1
                components["defensive_reward"] += self.confrontational_def_reward

            # Assessing player's positioning to encourage strategic defense and counterattack readiness
            if o['left_team_active'][o['active']]:
                own_goal = -1.0  # Assume own goal x-coordinate is -1
                player_pos = o['left_team'][o['active']][0]  # Player x-coordinate
                # Encouraging players to be positioned towards own half but ready for quick counters
                if own_goal + 0.5 < player_pos < own_goal + 0.15:
                    components["positional_play_reward"] += self.positional_play_reward
                    self.defensive_positions_achieved += 1

        reward += components["defensive_reward"] + components["positional_play_reward"]
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = components["base_score_reward"] + components["defensive_reward"] + components["positional_play_reward"]
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
