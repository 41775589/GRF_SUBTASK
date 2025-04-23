import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance learning of defensive and transitional strategies."""

    def __init__(self, env):
        super().__init__(env)
        # Parameters for defensive reward shaping
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zone_threshold = -0.5  # Threshold for considering agents in a defensive position
        self.defensive_position_reward = 0.01  # Reward for maintaining position in defensive region
        self.ball_clearance_reward = 0.3       # Reward for clearing the ball from the defensive third

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "ball_clearance_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for defensive position
            player_pos = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]
            if player_pos < self.defensive_zone_threshold:
                components["defensive_reward"][rew_index] += self.defensive_position_reward
            
            # Reward for clearing the ball
            if o['ball_owned_team'] == -1 and o['last_action'] == 'clear':
                components["ball_clearance_reward"][rew_index] += self.ball_clearance_reward

            # Aggregate the rewards
            reward[rew_index] += (components["defensive_reward"][rew_index] + 
                                  components["ball_clearance_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Append component values to info for transparency
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions counter for debugging or analytical purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info
