import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that promotes crossing and sprinting abilities for wingers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.crossing_rewards_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.crossing_rewards_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['crossing_rewards_collected'] = self.crossing_rewards_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.crossing_rewards_collected = from_pickle['crossing_rewards_collected']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            if o['sticky_actions'][8] and o['ball_owned_team'] == 1:
                team_direction = o['right_team_direction'][o['active']]
                # Check for sprinting towards the sides of the pitch
                if np.abs(team_direction[1]) > 0.1:
                    crossing_location = np.abs(o['ball'][0])
                    # Crossing near the opponent goal line
                    if crossing_location > 0.7:
                        if i not in self.crossing_rewards_collected:
                            self.crossing_rewards_collected[i] = True
                            # Additional reward for successful high-speed dribbling and positioning for crossing
                            components["crossing_reward"][i] = 0.5
                            reward[i] += components["crossing_reward"][i]
        
        # Reset sprint counters
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for j, action in enumerate(agent_obs['sticky_actions']):
                # Count sprint actions
                self.sticky_actions_counter[j] = max(self.sticky_actions_counter[j], action)
        
        reward = [initial_r + sprint_bonus if action_sprint else initial_r 
                  for initial_r, action_sprint, sprint_bonus in zip(
                      reward, self.sticky_actions_counter, components['crossing_reward'])]
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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
