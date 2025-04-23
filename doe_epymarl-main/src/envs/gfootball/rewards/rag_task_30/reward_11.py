import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic defensive positioning reward."""
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_transitions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_transitions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DefensiveTransitionWrapper'] = self.defensive_transitions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_transitions = from_pickle['DefensiveTransitionWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Looking for strategic defensive positioning transition
            if self._is_team_in_defensive_posture(o):
                if not self.defensive_transitions.get(rew_index):
                    components["defensive_transition_reward"][rew_index] = 0.3
                    reward[rew_index] += components["defensive_transition_reward"][rew_index]
                    self.defensive_transitions[rew_index] = True

        return reward, components

    def _is_team_in_defensive_posture(self, obs):
        """ Evaluate the team's strategic defensive positioning """
        players_positions = obs['left_team'] if obs['ball_owned_team'] == 1 else obs['right_team']
        threshold_y = np.median([player[1] for player in players_positions])
        team_has_defensive_lines = all(player[1] > threshold_y for player in players_positions)
        return team_has_defensive_lines

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
