import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that encourages effective mid to long-range passing in football games.
    This focuses on enhancing strategic use of high and long passes and rewards precision
    in passing from different regions of the pitch.
    """
    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for passes
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_regions = 5  # Define regions on the pitch for varying passing difficulty
        self.region_passes_collected = {}
        self.pass_precision_reward = 0.05
        self.high_long_pass_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.region_passes_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.region_passes_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.region_passes_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_precision_reward": [0.0] * len(reward),
                      "high_long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            ball_position = o['ball'][0]  # Use the x position for simplicity.
            ball_owned_team = o['ball_owned_team']

            # Only consider when the ball is owned by our team.
            if ball_owned_team == 0 or ball_owned_team == 1:
                region_index = min(int(abs(ball_position + 1) // (2 / self.passing_regions)), self.passing_regions - 1)

                # Check for a successful pass; for simplicity, assume it's when possession changes with ball advancement
                previous_passer = self.region_passes_collected.get(rew_index, {}).get('last_passer', None)
                current_player = o['ball_owned_player']
                player_team = o['ball_owned_team']

                if previous_passer is not None and previous_passer != current_player and player_team == o['designated']:
                    # Assuming higher difficulty in precision for farther regions
                    reward_increment = self.pass_precision_reward * (1 + region_index)
                    components["pass_precision_reward"][rew_index] = reward_increment
                    reward[rew_index] += reward_increment

                # Update the last passer if our team still has the ball
                self.region_passes_collected[rew_index] = {'last_passer': current_player}

                # Encourage high and long passes: Check if the action includes high power and 'long pass' strategy.
                # Example use of sticky_actions for 'action_long_pass' assumed as index 10 
                if o['sticky_actions'][10] and o['ball_direction'][2] > 0.1:  # Assume upward z direction indicates high pass
                    components["high_long_pass_reward"][rew_index] = self.high_long_pass_reward
                    reward[rew_index] += self.high_long_pass_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counter for debugging purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                if action_value:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action_value
        return observation, reward, done, info
