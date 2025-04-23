import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful high passes and crosses aimed at supporting dynamic attacking plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_checks_completed = {}
        self.num_pass_zones = 5  # Dividing the opponent's half into 5 horizontal zones
        self.pass_reward_increment = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_checks_completed = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.pass_checks_completed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_checks_completed = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] in {4, 5}:  # Considering only crosses or high passes
                # Evaluating the player's position on the y-axis
                y_pos = o['left_team'][o['active']][1] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][1]
                reward_change, zone = self.evaluate_pass(o['ball_owned_team'], y_pos)
                if reward_change and rew_index not in self.pass_checks_completed:
                    components["pass_reward"][rew_index] += self.pass_reward_increment
                    reward[rew_index] += components["pass_reward"][rew_index]
                    self.pass_checks_completed[rew_index] = zone
        return reward, components

    def evaluate_pass(self, team, y_pos):
        """ Determines if a cross or high pass from a given zone should be rewarded based on team side. """
        zone_boundaries = np.linspace(-0.42, 0.42, self.num_pass_zones+1)  # Y-axis boundaries for each zone
        for i in range(len(zone_boundaries)-1):
            if zone_boundaries[i] < y_pos <= zone_boundaries[i+1]:
                if team == 1 and i >= len(zone_boundaries) / 2:  # Right team crossing from left half
                    return True, i
                elif team == 0 and i < len(zone_boundaries) / 2:  # Left team crossing from right half
                    return True, i
        return False, None

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
