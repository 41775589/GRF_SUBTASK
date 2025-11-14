import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds advanced defensive and counterattack rewards for soccer agents."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Introducing individual rewards for defense and counter attacks
        self.defensive_zone_reward = 0.2
        self.counterattack_bonus = 0.3
        self.ball_recovery_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_zone_reward": [0.0] * len(reward),
            "counterattack_bonus": [0.0] * len(reward),
            "ball_recovery_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            base_reward = reward[rew_index]

            # Enhance reward for good defensive positions near own goal
            if o['left_team'][o['active']][0] < -0.7:  # Left team's active player in the defensive third
                components["defensive_zone_reward"][rew_index] = self.defensive_zone_reward
                base_reward += components["defensive_zone_reward"][rew_index]
            
            # Increase reward for recovering the ball and moving to counterattack
            if o['ball_owned_team'] == 0 and self.prev_ball_owned_team != 0:
                if o['left_team'][o['active']][0] > -0.3:  # Assuming the player has advanced toward the mid-field
                    components["counterattack_bonus"][rew_index] = self.counterattack_bonus
                    base_reward += components["counterattack_bonus"][rew_index]

            # Reward for recovering control of the ball at any point
            if self.prev_ball_owned_team == 1 and o['ball_owned_team'] == 0:
                components["ball_recovery_reward"][rew_index] = self.ball_recovery_reward
                base_reward += components["ball_recovery_reward"][rew_index]

            reward[rew_index] = base_reward
            self.prev_ball_owned_team = o['ball_owned_team']

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
