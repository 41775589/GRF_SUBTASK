import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that improves defending strategies by rewarding proficient tackling,
    efficient movement control, and effective passing under pressure.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to tune the importance of different components of the reward
        self.tackle_reward_coefficient = 2.0
        self.movement_efficiency_coefficient = 1.0
        self.pressured_pass_coefficient = 3.0

    def reset(self):
        """
        Reset the environment and sticky_actions_counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Retrieve the internal state for serialization.
        """
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the internal state from deserialization.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Compute the reward with additional components for tackling proficiency,
        movement efficiency, and pressured passing.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "movement_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]

            # Encouraging good tackles
            # Assuming tackles would lead to ball possession changes or prevent shots
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and 'ball_owned_player' not in o:
                components["tackle_reward"][rew_index] = self.tackle_reward_coefficient
                reward[rew_index] += components["tackle_reward"][rew_index]

            # Reward for efficient movement (positioning and reducing stamina use)
            if 'right_team_tired_factor' in o and o['right_team_tired_factor'][o['active']] < 0.1:
                components["movement_reward"][rew_index] = self.movement_efficiency_coefficient
                reward[rew_index] += components["movement_reward"][rew_index]

            # Advancing passing skills under pressure
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                # More reward if they are closer to opponent goal when passing
                ball_x = o['ball'][0]
                if ball_x > 0.5:  # Assuming right half is opponent's side
                    components["passing_reward"][rew_index] = self.pressured_pass_coefficient
                    reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take an action using the underlying env, augment the reward, and add diagnostics.
        """
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
