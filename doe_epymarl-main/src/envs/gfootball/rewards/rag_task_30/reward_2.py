import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a positioning and transition reward focused on defensive resilience and counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self._defensive_positions = {}
        self._transition_speed_bonus = {}
        self.positioning_reward = 0.05
        self.transition_bonus = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset the sticky actions and collected data on defensive positions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_positions = {}
        self._transition_speed_bonus = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the state of the defensive positions and transition bonuses
        to_pickle['DefensivePositions'] = self._defensive_positions
        to_pickle['TransitionSpeedBonus'] = self._transition_speed_bonus
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore the state of defensive positions and transition bonuses from pickle
        from_pickle = self.env.set_state(state)
        self._defensive_positions = from_pickle['DefensivePositions']
        self._transition_speed_bonus = from_pickle['TransitionSpeedBonus']
        return from_pickle

    def reward(self, reward):
        # Re-define rewards based on positioning and successful transitions
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "transition_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage strategic positioning defensively
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # Team 0 is our team's index
                if 'right_team' in o:
                    for player_idx, pos in enumerate(o['right_team']):
                        if pos[0] < 0 and player_idx == o['active']:  # Defensive half of the field
                            self._defensive_positions[rew_index] = self._defensive_positions.get(rew_index, 0) + 1
                            if self._defensive_positions[rew_index] == 1:  # Reward only the first time positioned well
                                reward[rew_index] += self.positioning_reward
                                components["positioning_reward"][rew_index] = self.positioning_reward
            
            # Encourage quick transition from defense to attack
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                if o.get('left_team_direction', np.zeros_like(o['left_team']))[o['active']][0] > 0.1:  # Fast movement towards opponent's goal
                    self._transition_speed_bonus[rew_index] = self._transition_speed_bonus.get(rew_index, 0) + 1
                    if self._transition_speed_bonus[rew_index] == 10:
                      reward[rew_index] += self.transition_bonus
                      components["transition_bonus"][rew_index] = self.transition_bonus
        
        return reward, components

    def step(self, action):
        # Follow the usual step but include the redefined reward and add component values to info
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
