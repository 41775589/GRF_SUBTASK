import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for defensive actions and strategic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.positional_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.positional_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_position_reward": [0.0] * len(reward),
                      "interception_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for strategic defensive positioning based on player and ball position.
            if o['ball_owned_team'] == 1:  # Ball is owned by the opposing team
                ball_pos = o['ball'][:2]  # Get x, y coordinates of the ball
                player_pos = o['left_team'][o['active']][:2]  # Get x, y coordinates of the active player
                distance_to_ball = np.linalg.norm(player_pos - ball_pos)
                
                # Defensive reward: strategic positioning if within a certain radius of ball
                if distance_to_ball < 0.15:
                    components["defensive_position_reward"][rew_index] = 0.1
                    reward[rew_index] += components["defensive_position_reward"][rew_index]

            # Reward for successful interceptions
            if o['game_mode'] == 2:  # Game mode corresponds to interception scenarios
                if 'interceptions' not in self.positional_rewards.get(rew_index, {}):
                    self.positional_rewards[rew_index] = {'interceptions': 0}
                self.positional_rewards[rew_index]['interceptions'] += 1
                components["interception_reward"][rew_index] = 0.2 * self.positional_rewards[rew_index]['interceptions']
                reward[rew_index] += components["interception_reward"][rew_index]

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
