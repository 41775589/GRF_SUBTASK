import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense crossing and sprinting ability reward focused on wingers."""

    def __init__(self, env):
        super().__init__(env)  # Initialize the gym RewardWrapper
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Retrieve the latest observation to assess the situation
        observation = self.env.unwrapped.observation()
        # Base Reward elements
        components = {"base_score_reward": reward.copy()}

        for i in range(len(reward)):
            # Extract necessary data for easier access
            o = observation[i]
            ball_pos = np.array(o['ball'][:2])  # Only consider x, y positions of the ball
            player_pos = o['right_team' if o['active'] > 4 else 'left_team'][o['active']]
            distance_to_ball = np.linalg.norm(ball_pos - player_pos)

            components.setdefault("crossing_ability_reward", [0.0] * len(reward))
            components.setdefault("sprinting_reward", [0.0] * len(reward))
            
            # Encourage sprinting by providing reward based on sticky actions[8] (sprint)
            if o['sticky_actions'][8] == 1:  # Check if the sprint action is active
                components["sprinting_reward"][i] += 0.05
            
            # Encourage crossing the ball from the wings (assumes positions close to y=-0.42 or y=0.42 are wings)
            if abs(player_pos[1]) > 0.35 and distance_to_ball < 0.1:
                components["crossing_ability_reward"][i] += 0.1
            
            # Update individual rewards
            reward[i] += components["sprinting_reward"][i] + components["crossing_ability_reward"][i]

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
