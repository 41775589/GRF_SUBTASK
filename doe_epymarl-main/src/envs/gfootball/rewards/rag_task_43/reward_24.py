import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive maneuvering and quick transition to counterattacks."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positioning_threshold = 0.1
        self.ball_recovery_reward = 0.2
        self.counterattack_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_positioning_reward": [0.0] * len(reward), "ball_recovery_reward": [0.0] * len(reward), "counterattack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Evaluate defensive positioning
            if o['game_mode'] == 3:  # If it's a Free Kick
                enemy_distance = np.min(np.linalg.norm(o['left_team'] - o['ball'], axis=1))
                if enemy_distance > self.defensive_positioning_threshold:
                    components["defensive_positioning_reward"][rew_index] = self.defensive_positioning_threshold
                    reward[rew_index] += components["defensive_positioning_reward"][rew_index]

            # Reward for ball recovery
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["ball_recovery_reward"][rew_index] = self.ball_recovery_reward
                reward[rew_index] += components["ball_recovery_reward"][rew_index]

            # Reward for initiating a counterattack
            if o['ball_owned_team'] == 0 and np.linalg.norm(o['ball_direction'][:2]) > 0.5:  # Assuming the ball is moving fast
                components["counterattack_reward"][rew_index] = self.counterattack_reward
                reward[rew_index] += components["counterattack_reward"][rew_index]

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
