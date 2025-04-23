import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for close-range attacks, shot precision, and dribble effectiveness."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_targeted_multiplier = 0.2
        self.dribble_efficiency_multiplier = 0.1

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_precision_reward": [0.0] * len(reward),
                      "dribble_efficiency_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for shot precision when close to the goal area
            if self.is_close_to_goal(o) and self.is_shot_on_target(o):
                components["shot_precision_reward"][rew_index] = self.shot_targeted_multiplier

            # Reward for effective dribble when possessing the ball
            if self.is_dribbling_efficient(o):
                components["dribble_efficiency_reward"][rew_index] = self.dribble_efficiency_multiplier

            # Calculate the final rewards for each agent
            reward[rew_index] += components["shot_precision_reward"][rew_index] + components["dribble_efficiency_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Aggregating the components values into info for monitoring
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Handle sticky actions counter
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info

    def is_close_to_goal(self, observation):
        """Determine if the player is close to the opponent's goal."""
        ball_pos = observation['ball'][0:2]
        goal_pos = [1, 0]  # Using constant goal X position, Y is approximately 0
        distance_to_goal = np.linalg.norm(np.array(ball_pos) - np.array(goal_pos))
        return distance_to_goal < 0.2

    def is_shot_on_target(self, observation):
        """Check if the agent attempted a shot towards the goal."""
        return observation['action'] == 9  # Assuming 9 corresponds to the shooting action

    def is_dribbling_efficient(self, observation):
        """Efficient dribbling when sticky action dribble is used at high ball possession."""
        return observation['sticky_actions'][9] == 1 and observation['ball_owned_team'] == 0  # Using hypothetical indexes
