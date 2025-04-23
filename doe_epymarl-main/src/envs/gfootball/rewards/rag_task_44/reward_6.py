import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper adding a reward for stop-dribbling under pressure."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_control = False
        self.stop_dribble_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_control = False
        return self.env.reset()

    def get_state(self, to_pickle):
        state_data = to_pickle['base'] = {}
        state_data['sticky_actions_counter'] = self.sticky_actions_counter
        state_data['previous_ball_control'] = self.previous_ball_control
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['base']
        self.sticky_actions_counter = state_data['sticky_actions_counter']
        self.previous_ball_control = state_data['previous_ball_control']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_dribble_reward": [0.0] * len(reward)}
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            has_control_now = (o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active'])
            just_stopped_dribbling = (self.previous_ball_control and not has_control_now and
                                      self.sticky_actions_counter[9] > 0)
            # Increment reward for successful stop-dribble under pressure
            if just_stopped_dribbling:
                components["stop_dribble_reward"][rew_index] = self.stop_dribble_reward
                reward[rew_index] += components["stop_dribble_reward"][rew_index]

            # Update previous control state
            self.previous_ball_control = has_control_now

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
