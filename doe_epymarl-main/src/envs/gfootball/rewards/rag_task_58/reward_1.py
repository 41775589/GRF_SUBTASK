import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for defensive coordination and transition to attack."""

    def __init__(self, env):
        super().__init__(env)
        self._num_defensive_actions = 5  # Number of defensive zones/actions
        self._defensive_rewards = np.zeros(self._num_defensive_actions)
        self._reward_for_defensive_action = 0.05
        self._reward_for_successful_transition = 0.1  # Reward for successful defense to attack transition
        self._previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_rewards.fill(0)
        self._previous_ball_owner = None
        return self.env.reset()

    def reward(self, reward):
        # Initialize components dictionary for detailed reward explanations
        components = {
            "base_score_reward": reward.copy(),
            "defensive_rewards": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            # Check if the opposing team (right) has possession
            if o['ball_owned_team'] == 1 and self._previous_ball_owner != 1:
                self._defensive_rewards.fill(0)  # Reset defensive rewards when possession changes to opponent
                self._previous_ball_owner = 1

            # Incremental reward based on defensive actions and zone control
            if o['ball_owned_team'] == 0 and self._previous_ball_owner == 1:
                zone_index = self._determine_zone(o['ball'])
                if zone_index is not None and self._defensive_rewards[zone_index] == 0:
                    self._defensive_rewards[zone_index] = 1
                    components["defensive_rewards"][i] += self._reward_for_defensive_action
                    reward[i] += self._reward_for_defensive_action

            # Transition reward
            if o['ball_owned_team'] == 0 and self._previous_ball_owner == 0 and reward[i] > 0:
                components["transition_reward"][i] += self._reward_for_successful_transition
                reward[i] += self._reward_for_successful_transition

        return reward, components

    def _determine_zone(self, ball_pos):
        # Simplistic approach: divide field width into defensive zones
        x = ball_pos[0]
        if -1.0 <= x <= -0.6:
            return 0
        elif -0.6 < x <= -0.2:
            return 1
        elif -0.2 < x <= 0.2:
            return 2
        elif 0.2 < x <= 0.6:
            return 3
        elif 0.6 < x <= 1.0:
            return 4
        return None

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
