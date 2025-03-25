import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward specifically for the role of a 'sweeper' in football.
    This role is designed to effectively clear the ball from dangerous areas, make critical tackles,
    and provide fast recoveries, particularly in the defensive zone.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward),
                      "recovery_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            ball_dist = ((o['ball'][0] - o['left_team'][o['active']][0]) ** 2 +
                         (o['ball'][1] - o['left_team'][o['active']][1]) ** 2) ** 0.5
            
            # Reward for clearing the ball from close proximity to own goal
            if o['ball_owned_team'] == 1 and o['ball'][0] < -0.5:
                components['clearance_reward'][rew_index] = 1.0
            reward[rew_index] += components['clearance_reward'][rew_index]

            # Reward for making a tackle
            if o['game_mode'] == 3 and o['left_team_roles'][o['active']] == 9:
                components['tackle_reward'][rew_index] = 0.5
            reward[rew_index] += components['tackle_reward'][rew_index]

            # Reward for quick recovery i.e., returning ball possession under pressure
            if o['ball_owned_team'] == 0 and ball_dist < 0.1:
                components['recovery_reward'][rew_index] = 0.3
            reward[rew_index] += components['recovery_reward'][rew_index]

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
