import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering midfield dynamics and strategic repositioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_threshold = 0.2  # Defines the midfield territory
        self.ball_control_reward = 0.05
        self.transition_reward = 0.1
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
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_control": [0.0] * len(reward),
            "transition": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            teams_midfield = abs(o['ball'][0]) < self.midfield_threshold

            if teams_midfield:
                if o['ball_owned_team'] == 0:  # Our agent's team owns the ball
                    components["ball_control"][rew_index] = self.ball_control_reward
                    reward[rew_index] += components["ball_control"][rew_index]
                
            transition_phase = o.get('game_mode', 0) in (1, 2, 3, 4, 5, 6)  # Transition game modes
            if transition_phase:
                components["transition"][rew_index] = self.transition_reward
                reward[rew_index] += components["transition"][rew_index]

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
            for i, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
