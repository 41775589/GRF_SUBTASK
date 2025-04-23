import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward focusing on enhancing team's mid to long-range passing effectiveness.
    The reward encourages precise, strategic high and long passes between players, oriented towards gameplay development.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define rewards for successful high and long passes
        self.successful_pass_reward = 0.2

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Initialize the component for each agent
            components["passing_reward"][rew_index] = 0.0

            # Check if a pass has been performed by analyzing ball transfer between players
            if ('ball_owned_team' in o and o['ball_owned_team'] in [0, 1] and 
                    'ball_owned_player' in o and o['ball_owned_player'] >= 0):
                player_id = o['active']
                owning_player_id = o['ball_owned_player']
                # Identify long and high passes from the ball trajectory
                ball_dist = np.linalg.norm(o['ball_direction'])
                if ball_dist > 0.30:  # threshold for long passes
                    # Assess the quality of the pass (e.g., did it maintain control?)
                    # Currently simplifying to always successful for demonstration
                    components["passing_reward"][rew_index] = self.successful_pass_reward
                    reward[rew_index] += components["passing_reward"][rew_index]

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
