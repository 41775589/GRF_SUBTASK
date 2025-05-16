import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper focusing on mastering passes from defensive areas under pressure using 'Short Pass' and 'High Pass'."""
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.short_pass_reward = 0.3
        self.high_pass_reward = 0.5
        self.pressure_threshold = 0.2
        self.defensive_zone = -0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "short_pass": [0.0], "high_pass": [0.0]}

        if observation is None:
            return reward, components

        o = observation[0]  # Since number of agents is 1

        # Ensure ball possession by our agent and in the defensive zone
        if o['ball_owned_team'] == 0 and o['left_team'][o['active']][0] < self.defensive_zone:
            ball_pos = o['ball'][:2]
            player_pos = o['left_team'][o['active']]

            # Calculate pressure based on proximity of any opponent in the vicinity
            opponents_pos = o['right_team']
            pressure = any([np.linalg.norm(player_pos - opp) < self.pressure_threshold for opp in opponents_pos])

            if pressure:
                # Check for 'short pass' in action
                if o['sticky_actions'][6]:    # Assuming the index for 'Short Pass'
                    components["short_pass"][0] = self.short_pass_reward
                    reward[0] += components["short_pass"][0]

                # Check for 'high pass' in action
                if o['sticky_actions'][7]:    # Assuming the index for 'High Pass'
                    components["high_pass"][0] = self.high_pass_reward
                    reward[0] += components["high_pass"][0]

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
