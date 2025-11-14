import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for proficient use of Short Pass and Long Pass in high-risk areas."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Dynamically calculates reward based on control of the ball and successful short or long passes in high-risk areas."""
        components = {"base_score_reward": reward.copy(), "high_risk_pass_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the agent performed a pass in a high-risk area
            if o['active'] != -1 and o['ball_owned_player'] == o['active']:
                if o['ball_owned_team'] == 0:  # Assuming 0 is the index for the controlled team
                    ball_position = np.array(o['ball'][:2])

                    # Define high-risk zones roughly as being close to own goal
                    own_goal_position = np.array([-1, 0])  # Simulation of own goal position
                    distance_to_own_goal = np.linalg.norm(ball_position - own_goal_position)

                    if distance_to_own_goal < 0.3:  # Arbitrary threshold for high-risk zones
                        if o['sticky_actions'][6] or o['sticky_actions'][5]:  # Indices for long pass and short pass actions
                            # Add a substantial reward for passing in high-risk zones to encourage clearing the ball
                            components["high_risk_pass_reward"][rew_index] = 0.5

        reward = np.add(reward, components["high_risk_pass_reward"])
        return reward.tolist(), components

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
