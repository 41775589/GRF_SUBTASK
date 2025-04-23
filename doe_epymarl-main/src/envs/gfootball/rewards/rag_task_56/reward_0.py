import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom wrapper to enhance team defensive capabilities primarily focusing on goalkeeper's shot-stopping and defenders' tackling and ball retention."""
    def __init__(self, env):
        super().__init__(env)
        self._defensive_efficiency = 0.1  # Coefficient for defensive actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset sticky actions counter and other member data for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get wrapper's current state regarding checkpoints."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state from the saved state."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modify the rewards given the observations tailored to enhance defensive capabilities."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["defensive_reward"][rew_index] = 0.0

            # Reward for goalkeeper involvement in shot-stopping
            if o['right_team_roles'][o['active']] == 0:  # Assuming 0 is the role index for goalkeeper
                if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                    if np.linalg.norm(o['ball_direction']) > 0.1:
                        components["defensive_reward"][rew_index] = self._defensive_efficiency

            # Reward for defenders' tackling and effective ball retention
            if o['right_team_roles'][o['active']] in [1, 2, 3, 4]:  # Assuming these are defender roles
                if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                    components["defensive_reward"][rew_index] = self._defensive_efficiency

            reward[rew_index] += components["defensive_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Environment step with additional information processing in rewards."""
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
