import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic offensive maneuver reward specific to quick attacks and game phase transitions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.quick_attack_reward = 0.5  # Reward for quick constructive actions
        self.game_phase_transition_reward = 0.3  # Reward for adaptively changing play during different game phases

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Reset any necessary state here from from_pickle if any.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "quick_attack_reward": [0.0] * len(reward),
            "game_phase_transition_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check for quick offensive actions (quick forward passes or sprints with ball).
            if o['sticky_actions'][8] == 1 and o['ball_owned_team'] == 0:  # Sprint action with ball
                components["quick_attack_reward"][rew_index] = self.quick_attack_reward
            if o['game_mode'] in np.arange(1, 7) and o['ball_owned_team'] == 0:  # Adaption to game phase, excluding normal play
                # When game mode changes (corner, throwin etc.), rewards are given if the team adapts by taking initiative.
                components["game_phase_transition_reward"][rew_index] = self.game_phase_transition_reward

            # Calculate the total reward with extra components
            reward[rew_index] += components["quick_attack_reward"][rew_index] + components["game_phase_transition_reward"][rew_index]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
