import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive gameplay reward focused on interceptions and defensive positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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

        components = {
            "base_score_reward": reward.copy(),
            "defensive_positioning_reward": [0.0] * len(reward),
            "interception_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                # Reward based on the team having possession
                components["defensive_positioning_reward"][rew_index] = 0.02  # Small reward for possession

            # Evaluate interception potential
            if ('ball_owned_team' in o and o['ball_owned_team'] != o['active']
                    and 'left_team' in o and 'right_team' in o):
                enemy_team_key = 'right_team' if o['ball_owned_team'] == 1 else 'left_team'
                for enemy in o[enemy_team_key]:
                    # Reward proximity to enemy players without the ball
                    distance = np.linalg.norm(np.array(o['left_team'][o['active']]) - enemy)
                    if distance < 0.1:  # If within a close range to intercept
                        components["interception_reward"][rew_index] += 0.1

        for idx, (base, def_pos, intercept) in enumerate(zip(reward, components['defensive_positioning_reward'], components['interception_reward'])):
            reward[idx] = base + def_pos + intercept
            
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
