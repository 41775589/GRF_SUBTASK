import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for effective clearance under pressure in defensive zones."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._clearance_success_reward = 0.5
        self._clearance_attempt_penalty = -0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_success_reward": [0.0] * len(reward),
                      "clearance_attempt_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # In defensive mode and under pressure
            if o['game_mode'] in {2, 3, 4, 5, 6} and o['ball_owned_player'] == o['active']:
                # Defensive clearance
                ball_pos_x = o['ball'][0]
                if ball_pos_x > -0.5 and ball_pos_x < 0:  # Ball in own half and defensive area
                    if np.linalg.norm(o['ball_direction'][:2]) > 0:  # Ball is moving
                        components["clearance_success_reward"][rew_index] = self._clearance_success_reward
                        reward[rew_index] += components["clearance_success_reward"][rew_index]
                    else:
                        components["clearance_attempt_penalty"][rew_index] = self._clearance_attempt_penalty
                        reward[rew_index] += components["clearance_attempt_penalty"][rew_index]

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
