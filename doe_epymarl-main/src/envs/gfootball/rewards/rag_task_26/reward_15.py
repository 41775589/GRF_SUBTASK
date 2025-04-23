import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides specific rewards based on midfield dynamics and roles."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_rewards = {}
        self.setup_midfield_rewards()

    def setup_midfield_rewards(self):
        """Setup custom rewards for better midfield control based on specific conditions."""
        self.midfield_control_rewards = {
            'central_control': 0.1,  # Central players contributing to control
            'wide_support': 0.05,    # Wide midfielders supporting effectively
            'defensive_actions': 0.2, # Effective defensive maneuvers in midfield
            'offensive_support': 0.15 # Support to forward movements
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_control_rewards'] = self.midfield_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control_rewards = from_pickle.get('midfield_control_rewards', self.midfield_control_rewards)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball'][0]  # x-coordinate of the ball

            # Check for central midfield control
            if (o['left_team_roles'][o['active']] == 5 or  # Central midfield
                o['right_team_roles'][o['active']] == 5):  # Central midfield
                if abs(ball_position) < 0.3:  # Ball is roughly centrally located
                    components['central_control'] = self.midfield_control_rewards['central_control']
                    reward[rew_index] += components['central_control']

            # Check for wide support
            if (o['left_team_roles'][o['active']] == 6 or  # Left midfield
                o['right_team_roles'][o['active']] == 7):  # Right midfield
                if abs(ball_position) > 0.5: # Ball is located on the wings
                    components['wide_support'] = self.midfield_control_rewards['wide_support']
                    reward[rew_index] += components['wide_support']

            # Adding rewards based on effective defensive manuevers
            if 'defensive_actions' in o:
                components['defensive_actions'] = self.midfield_control_rewards['defensive_actions']
                reward[rew_index] += components['defensive_actions']

            # Support for forward movements
            if 'offensive_support' in o:
                components['offensive_support'] = self.midfield_control_rewards['offensive_support']
                reward[rew_index] += components['offensive_support']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
