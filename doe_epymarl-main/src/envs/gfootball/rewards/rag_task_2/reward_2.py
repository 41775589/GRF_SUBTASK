import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive strategy and coordination."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_coordination": [0.0] * len(reward)
        }

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage ball recovery and maintaining formations to maximize coverage
            if o["ball_owned_team"] == 0:  # If ball is owned by team 0 (defensive team in this context)
                # Reward for keeping the ball in defensive control
                components["defensive_coordination"][rew_index] += 0.05
                reward[rew_index] += components["defensive_coordination"][rew_index]
                # Reward decreasing the distance to defensive zones when in possession
                opponent_goal_dist = 1 - o['ball'][0]  # Normalize based on field dimensions
                components["defensive_coordination"][rew_index] += 0.05 * opponent_goal_dist
                reward[rew_index] += components["defensive_coordination"][rew_index]

            # More defensive players between ball and own goal boosts the reward
            # Count players between the ball and goal
            own_goal_x = -1
            players_defending = sum([
                1 for pos in o['left_team'] if (pos[0] < o['ball'][0] < own_goal_x) or (o['ball'][0] < pos[0] < own_goal_x)
            ])
            components["defensive_coordination"][rew_index] += 0.01 * players_defending
            reward[rew_index] += components["defensive_coordination"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        observation = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
