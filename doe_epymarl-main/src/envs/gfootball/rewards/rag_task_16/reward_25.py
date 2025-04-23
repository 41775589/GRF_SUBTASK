import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages high pass execution with precision."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_accuracy_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            ball_z_pos = o['ball'][2]
            ball_direction = o['ball_direction'][2]

            # High pass execution: Assuming high_pass as actions involving large z in ball_direction and ball height
            if ball_direction > 0.2 and ball_z_pos > 0.15:
                # Adding precision evaluation by considering completion of the pass
                if o['ball_owned_team'] == o['active'] and o['game_mode'] == 0:
                    # Reward adjustment based on precision and completion of the high pass
                    components["high_pass_accuracy_reward"][i] = self.high_pass_accuracy_reward
                    reward[i] += 1.5 * components["high_pass_accuracy_reward"][i]

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
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
