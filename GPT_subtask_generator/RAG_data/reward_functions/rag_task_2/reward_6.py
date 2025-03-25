import gym
import numpy as np
class CoordinationRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds an auxiliary reward for defensive strategies and coordination between agents."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CoordinationRewardWrapper'] = self._defensive_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._defensive_rewards = from_pickle['CoordinationRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Checking the state of ball possession and defensive positioning
            if o['ball_owned_team'] == 0 and o['right_team'][rew_index][0] > 0.5:
                # Check if there are multiple players from the team near the ball for enhanced defense
                team_position_average = np.mean([np.linalg.norm(o['right_team'][i] - o['ball'][:2]) 
                                                 for i in range(len(o['right_team']))])
                if team_position_average < 0.1:
                    # Reward for keeping the ball in the team with good defensive spread
                    components["defensive_reward"][rew_index] = 0.05 * (1 - team_position_average)
                    reward[rew_index] += components["defensive_reward"][rew_index]
        
            # Encourage, in general, moving the ball away from own goal
            ball_distance_from_own_goal = abs(o['ball'][0] + 1)
            components["defensive_reward"][rew_index] += 0.01 * ball_distance_from_own_goal
            reward[rew_index] += components["defensive_reward"][rew_index]

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
