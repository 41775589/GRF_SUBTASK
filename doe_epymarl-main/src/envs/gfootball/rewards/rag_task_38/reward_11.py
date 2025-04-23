import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for initiating counterattacks via accurate long passes and quick transitions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_completed = [False, False]  # Tracking if the long pass completion reward has been given
        # Coefficients
        self.long_pass_coefficient = 1.0
        self.transition_coefficient = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_completed = [False, False]
        return self.env.reset()

    def get_state(self, to_pickle):
        # Serialize any additional state variables specific to the custom reward here
        to_pickle['CheckpointRewardWrapper'] = {
            'long_pass_completed': self.long_pass_completed
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        stored = from_pickle.get('CheckpointRewardWrapper', {})
        self.long_pass_completed = stored.get('long_pass_completed', [False, False])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0, 0.0],
                      "transition_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            own_team = 'left_team' if o['ball_owned_team'] == 0 else 'right_team'
            opponent_team = 'right_team' if o['ball_owned_team'] == 0 else 'left_team'

            # Check for a successful long pass; criteria: long forward ball movement with change in ball ownership
            if o['ball_owned_team'] == rew_index and not self.long_pass_completed[rew_index]:
                if np.linalg.norm(o['ball_direction'][:2]) > 0.1 and o['ball'][0] * (-1 if own_team == 'left_team' else 1) > 0.5:
                    components["long_pass_reward"][rew_index] = self.long_pass_coefficient
                    self.long_pass_completed[rew_index] = True

            # Reward transition from defense to attack; criteria: Ball owns by own player and moving forward rapidly
            if o['ball_owned_team'] == rew_index:
                player_base_pos = o[own_team][o['active']]
                if player_base_pos[0] * (-1 if own_team == 'left_team' else 1) < -0.3:
                    components["transition_reward"][rew_index] = self.transition_coefficient

            # Incorporate calculated rewards
            reward[rew_index] += components["long_pass_reward"][rew_index] + components["transition_reward"][rew_index]

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
                self.sticky_actions_counter[i] = max(self.sticky_actions_counter[i], action)
        return observation, reward, done, info
