import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive teamwork strategy by tracking skilled positioning and control."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Adjusts the reward by adding bonus for defensive strategy and ball control."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Enhance reward based on defensive positioning and teamwork
        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            # Encourage maintaining control while being strategically positioned defensively
            if o['ball_owned_team'] == 0:  # Assuming '0' corresponds to controlled team
                ball_dist_to_goal = np.linalg.norm(o['ball'][:2] - [1, 0])  # Distance from the right goal
                player_pos = o['left_team'][o['active']]

                # Reward positioning closer to own goal when possessing the ball
                if ball_dist_to_goal > np.linalg.norm(player_pos[:2] - [1, 0]):
                    defensive_bonus = 0.05
                    components["defensive_positioning_reward"][rew_index] = defensive_bonus
                    reward[rew_index] += defensive_bonus

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

# Usage example with Env as Google Football environment
# env = FootballEnv(config)
# env = CheckpointRewardWrapper(env)
# obs = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # Replace with your agent's action
#     obs, reward, done, info = env.step(action)
