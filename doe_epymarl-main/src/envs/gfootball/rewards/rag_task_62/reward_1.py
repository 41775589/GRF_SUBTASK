import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on maximizing shooting angles and timing,
    specifically under high-pressure scenarios near the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_shots = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_shots = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['collected_shots'] = self.collected_shots
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_shots = from_pickle['collected_shots']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        # Strategy for reward function
        shooting_reward_scale = 1.0  # adjust the scale for shooting reward

        # Iterate through each agent's observation
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage shots under high pressure near the goal
            if 'ball_owned_by_team' in o and o['ball_owned_by_team'] == 1:  # Assuming the agent's team is 1
                ball_distance_to_goal = np.abs(o['ball'][0] - 1)  # x-distance to goal
                player_position_x = o['right_team'][o['active']][0]  # x-position of active player

                # Targets shots that are near the goal area
                if ball_distance_to_goal <= 0.2 and player_position_x > 0.75:
                    if not self.collected_shots.get(rew_index, False):
                        self.collected_shots[rew_index] = True
                        # Assign a shooting reward for appropriate shots
                        components["shooting_reward"][rew_index] = shooting_reward_scale
                        reward[rew_index] += components["shooting_reward"][rew_index]

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
