import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a goalkeeper coordination reward focusing on positioning and efficient clearances.
    This specific reward encourages goalkeeper behaviors that handle high-pressure scenarios by efficiently positioning 
    and clearing the ball to specific outfield players.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Initialize counting variables for goalkeeper actions during high-pressure scenarios
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_count = 0
        self.positioning_rewards = 0

    def reset(self):
        # Reset the counters for new episodes
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_count = 0
        self.positioning_rewards = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        # State to pickle includes the counts for positioning and clearances
        to_pickle['clearance_count'] = self.clearance_count
        to_pickle['positioning_rewards'] = self.positioning_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Retrieve state, including counts from the pickle
        from_pickle = self.env.set_state(state)
        self.clearance_count = from_pickle['clearance_count']
        self.positioning_rewards = from_pickle['positioning_rewards']
        return from_pickle

    def reward(self, reward):
        # Modify the reward by including positioning and clearance rewards
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward goalkeeper positioning
            if o['ball_owned_team'] == 0 and o['ball'][2] > 0.1 and 'active' in o and o['left_team_roles'][o['active']] == 0:
                # This assumes that the goalkeeper role index is 0
                components["positioning_reward"][rew_index] = 1.0
                reward[rew_index] += components["positioning_reward"][rew_index]
                self.positioning_rewards += 1

            # Reward clearances – aiming actions that move the ball towards outlying midfielders or attackers
            if o.get('game_mode', 0) in [2, 3, 4] and o['ball_direction'][0] > 0.5:
                # Checking for a strong forward movement during defensive game modes
                components["clearance_reward"][rew_index] = 5.0
                reward[rew_index] += components["clearance_reward"][rew_index]
                self.clearance_count += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Accumulate additional information about reward components into the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Add sticky action observations to info dictionary
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, sticky_action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += sticky_action
                info[f"sticky_actions_{i}"] = sticky_action

        return observation, reward, done, info
