import gym
import numpy as np
class GoalkeeperTrainingRewardWrapper(gym.RewardWrapper):
    """A wrapper designed specifically for goalkeeper training with enhanced reward signals."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_position_at_start = None
        self.goalkeeper_reflex_reward = 0.2
        self.counter_attack_initiation_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        observation = self.env.reset()
        self.goalkeeper_position_at_start = observation[0]['left_team'][0]
        return observation

    def get_state(self, to_pickle):
        to_pickle['GoalkeeperTrainingRewardWrapper'] = self.goalkeeper_position_at_start
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_position_at_start = from_pickle['GoalkeeperTrainingRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_reflex_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            goalie_pos = obs['left_team'][0]

            # Apply reflexes reward based on change in goalkeeper position indicating movement to stop a shot
            if np.linalg.norm(self.goalkeeper_position_at_start - goalie_pos) > 0.05:
                components["goalkeeper_reflex_reward"][rew_index] = self.goalkeeper_reflex_reward
                reward[rew_index] += self.goalkeeper_reflex_reward

            # Encourage playing effective long passes
            ball_pos = obs['ball']
            ball_owned_team = obs.get('ball_owned_team', -1)
            if ball_owned_team == 0:  # If left team owns the ball
                if np.linalg.norm(ball_pos[0:2] - goalie_pos) < 0.1 and obs.get('sticky_actions', [0])[7]:  # Check for 'long pass' action
                    components["counter_attack_reward"][rew_index] = self.counter_attack_initiation_reward
                    reward[rew_index] += self.counter_attack_initiation_reward

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
