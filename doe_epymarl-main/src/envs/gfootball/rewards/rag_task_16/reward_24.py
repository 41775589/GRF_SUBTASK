import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high passes with precision."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pass_quality_reward = 0.5  # Reward for each quality high pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        self.env.set_state(state)
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            # Check if the ball is owned by the team of this agent and the action was a high pass
            if o['ball_owned_team'] == 0 and self.env.unwrapped.action_space._last_action == 'high_pass':
                # Compute the quality of the high pass
                ball_z = o['ball'][2]  # Ball's z-axis position, higher values indicate higher passes
                if ball_z > 0.15:  # Assuming 0.15 is a threshold for considering a pass as 'high'
                    components["high_pass_reward"][i] = self._pass_quality_reward
                    reward[i] += components["high_pass_reward"][i]

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
                self.sticky_actions_counter[i] = action or self.sticky_actions_counter[i]
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
