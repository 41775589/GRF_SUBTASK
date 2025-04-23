import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on strategic positioning and transition from defense to counterattack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_checkpoints = {}
        self.reward_for_positioning = 0.05
        self.reward_for_transition = 0.1
        # Observations regarding the positioning thresholds
        self.defensive_zone = -0.5  # Defensive zone threshold on x-coordinate
        self.counterattack_zone = 0.3  # Threshold to enter opponent's half for transition
        self.last_ball_position = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_checkpoints = {}
        self.last_ball_position = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'position_checkpoints': self.position_checkpoints,
            'last_ball_position': self.last_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle.get('CheckpointRewardWrapper', {})
        self.position_checkpoints = wrapper_state.get('position_checkpoints', {})
        self.last_ball_position = wrapper_state.get('last_ball_position', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'positioning_reward': [0.0] * len(reward),
                      'transition_reward': [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward defensive positioning if within the defensive zone
            if o['active'] == o['designated'] and o['ball'][0] < self.defensive_zone:
                if not self.position_checkpoints.get(rew_index, {}).get('defended', False):
                    components['positioning_reward'][rew_index] = self.reward_for_positioning
                    self.position_checkpoints.setdefault(rew_index, {})['defended'] = True

            # Reward for transitioning to counterattack
            if o['ball_owned_team'] == 0 or o['ball_owned_team'] == 1:  # Check if either team owns the ball
                if self.last_ball_position < self.defensive_zone and o['ball'][0] > self.counterattack_zone:
                    components['transition_reward'][rew_index] = self.reward_for_transition

            # Update rewards
            reward[rew_index] += components['positioning_reward'][rew_index] + components['transition_reward'][rew_index]

            # Keep track of last ball position
            self.last_ball_position = o['ball'][0]

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
