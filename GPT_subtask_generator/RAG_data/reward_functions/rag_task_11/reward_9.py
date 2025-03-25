import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward based on offensive maneuvers and finishing control in football."""

    def __init__(self, env):
        super().__init__(env)
        self.num_zones = 5  # Number of zones towards the opponent's goal
        self.distance_reward = 0.05
        self.finish_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "offensive_reward": [0.0] * len(reward)}
        
        for rew_index, o in enumerate(observation):
            # Position of the ball on the x-axis, normalized between 0 and 1
            # Assume the left goal is at x = 0 and the right goal is at x = 1
            ball_position = o['ball'][0]  # Normalized position, 0 is left, 1 is right
            
            # Focus on the ball controlled by the agent's team
            if o['ball_owned_team'] == 1:  # Assuming agent's team is the right team
                # Calculate which zone the ball is in
                zone = int(ball_position * self.num_zones)
                # Reward for advancing the ball towards goal
                components['offensive_reward'][rew_index] = zone * self.distance_reward

                # Additional reward for finishing near the goal area
                if o['game_mode'] == 6 and o['ball'][1] < 0.05 and o['ball'][1] > -0.05:
                    # In the penalty area roughly
                    components['offensive_reward'][rew_index] += self.finish_reward

            # Update global reward with new component
            reward[rew_index] += components['offensive_reward'][rew_index]

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
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
