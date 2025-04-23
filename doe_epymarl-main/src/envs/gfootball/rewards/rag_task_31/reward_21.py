import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies original rewards to emphasize aggressive defense including tackles and slides."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize counters for special actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.3   # Reward given for successful tackle actions
        self.slide_reward = 0.5    # Reward given for successful slide actions
        self.defensive_position_bonus = 0.1   # Bonus for maintaining a good defensive position

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'tackle_reward': self.tackle_reward, 'slide_reward': self.slide_reward}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_reward = from_pickle['CheckpointRewardWrapper']['tackle_reward']
        self.slide_reward = from_pickle['CheckpointRewardWrapper']['slide_reward']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "slide_reward": [0.0] * len(reward),
                      "defensive_position_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Defensive actions: Tackles and Slides
            action_tackle = o['sticky_actions'][7]  # Assuming tackle is mapped to index 7
            action_slide = o['sticky_actions'][8]  # Assuming slide is mapped to index 8

            # Reward defensive actions if executed correctly
            if action_tackle:
                components["tackle_reward"][rew_index] = self.tackle_reward
            if action_slide:
                components["slide_reward"][rew_index] = self.slide_reward
            
            # Assume the defensive position bonus depends on player being close to his team's goal
            player_pos = o['left_team'][o['active']]  # active player's position
            goal_pos = [-1, 0]  # Assuming the goal position is constant
            distance_to_goal = np.sqrt((player_pos[0] - goal_pos[0])**2 + (player_pos[1] - goal_pos[1])**2)

            if distance_to_goal < 0.3:  # If player is within defensive radius
                components["defensive_position_bonus"][rew_index] += self.defensive_position_bonus

            # Compute the final modified reward
            reward[rew_index] += (components["tackle_reward"][rew_index] +
                                  components["slide_reward"][rew_index] +
                                  components["defensive_position_bonus"][rew_index])

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
