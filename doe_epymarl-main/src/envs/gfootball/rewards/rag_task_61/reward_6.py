import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper designed to promote team synergy during possession changes,
    and to emphasize strategic positioning and timing for both offensive and defensive moves.
    It aims to create incentives for intelligent plays relating to ball possession changes,
    positioning relative to teammates and opponents, and shaping future play opportunities.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_position_history = []
        self.loss_of_possession_penalty = -0.1
        self.successful_defense_bonus = 0.1
        self.team_formations_bonus = 0.05
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_position_history = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.player_position_history
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.player_position_history = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_penalty": [0.0] * len(reward),
                      "defensive_play_bonus": [0.0] * len(reward),
                      "formation_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, r in enumerate(reward):
            o = observation[rew_index]
            # Punish loss of possession
            if o['ball_owned_team'] != 1 and o['ball_owned_team'] != -1:
                components['possession_change_penalty'][rew_index] = self.loss_of_possession_penalty
                reward[rew_index] += components['possession_change_penalty'][rew_index]

            # Reward for good defensive positioning (e.g. intercepting the ball)
            if o['ball_owned_team'] == 1:  # Assuming '1' is our agent's team
                # Check if defending player has switched to the player marking the ball carrier
                if self.ball_is_close_to_opposing_player(o):
                    components['defensive_play_bonus'][rew_index] = self.successful_defense_bonus
                    reward[rew_index] += components['defensive_play_bonus'][rew_index]

            # Bonus for maintaining good team formations
            if self.is_good_formation(o):
                components['formation_bonus'][rew_index] = self.team_formations_bonus
                reward[rew_index] += components['formation_bonus'][rew_index]

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
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info

    def ball_is_close_to_opposing_player(self, observation):
        # Placeholder function to detect if ball is near an opposing player
        return False
    
    def is_good_formation(self, observation):
        # Placeholder function to evaluate if team is holding a strategic formation
        return True
