import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents for effective counterattacks through precise long passes and quick transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward factors
        self.long_pass_reward = 0.5
        self.quick_transition_reward = 1.0
        self.allowed_transition_steps = 10  # Hypothetical number of steps to consider a 'quick' transition

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle.get('CheckpointRewardWrapper', []), dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "quick_transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if a long pass just occurred
            if 'ball_direction' in o:
                # Hypothetical calculation for triggering a long pass reward
                ball_speed = np.linalg.norm(o['ball_direction'])
                if ball_speed > 0.5:  # Assume a threshold speed to determine a long pass
                    components["long_pass_reward"][rew_index] = self.long_pass_reward
                    reward[rew_index] += self.long_pass_reward

            # Check for quick transition from defense to attack
            if 'game_mode' in o:
                # Transition identified by a change from defensive mode (e.g., free kick in own half) to general play
                if o['game_mode'] == 3 and o['ball'][0] > 0 and self.env.unwrapped.steps_left > self.allowed_transition_steps:
                    components["quick_transition_reward"][rew_index] = self.quick_transition_reward
                    reward[rew_index] += self.quick_transition_reward

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
