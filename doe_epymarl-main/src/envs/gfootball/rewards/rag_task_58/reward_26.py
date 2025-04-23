import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward based on defensive actions and conversion to offensive play,
    specifically designed to promote advanced defensive coordination and effective counter-attacks.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define some thresholds and values for defensive rewards
        self.ball_recovery_reward = 0.3
        self.effective_pass_reward = 0.2
        self.defensive_positioning_reward = 0.1
        self.prev_ball_owned_team = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_owned_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state_info = self.env.get_state(to_pickle)
        state_info['prev_ball_owned_team'] = self.prev_ball_owned_team
        return state_info

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.prev_ball_owned_team = from_pickle.get('prev_ball_owned_team', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward),
                      "ball_recovery_reward": [0.0] * len(reward),
                      "effective_pass_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            if self.prev_ball_owned_team is not None and self.prev_ball_owned_team != o['ball_owned_team']:
                if o['ball_owned_team'] == 0:  # Ball recovered by the left team
                    components["ball_recovery_reward"][rew_index] = self.ball_recovery_reward
                    reward[rew_index] += components["ball_recovery_reward"][rew_index]

            # Encouraging maintaining formation and defensive positioning
            if o['ball_owned_team'] == -1:  # No team owns the ball
                components["defensive_positioning_reward"][rew_index] = self.defensive_positioning_reward
                reward[rew_index] += components["defensive_positioning_reward"][rew_index]

            # Reward for making effective passes that transition from defense to attack
            if o['ball_owned_team'] == 0 and 'action' in o and o['action'] == 'action_long_pass':
                components["effective_pass_reward"][rew_index] = self.effective_pass_reward
                reward[rew_index] += components["effective_pass_reward"][rew_index]

            self.prev_ball_owned_team = o['ball_owned_team']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions usage stats
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
