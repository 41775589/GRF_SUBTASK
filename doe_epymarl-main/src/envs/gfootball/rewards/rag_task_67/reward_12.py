import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a new type of reward that encourages ball control and passing during the transition from defense to attack."""

    def __init__(self, env):
        super().__init__(env)
        self.transition_rewards_collected = {}
        self.pass_success = 0.1
        self.continuous_control = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_rewards_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.transition_rewards_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.transition_rewards_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_success_reward": [0.0] * len(reward),
                      "continuous_control_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            components["pass_success_reward"][rew_index] = 0
            components["continuous_control_reward"][rew_index] = 0

            if o['game_mode'] in [2, 3, 4, 5, 6]:  # Not in normal play
                continue

            is_ball_owned_by_team = o['ball_owned_team'] == 0
            is_active_player_owns_ball = o['ball_owned_player'] == o['active']

            if is_ball_owned_by_team and is_active_player_owns_ball:
                # Reward short passes and ball control under pressure
                if self.sticky_actions_counter[8] == 1:  # sprint action
                    components["continuous_control_reward"][rew_index] += self.continuous_control
                if self.sticky_actions_counter[9] == 1:  # dribble action
                    components["continuous_control_reward"][rew_index] += self.continuous_control

                transition_key = (rew_index, "ball_control")
                if transition_key not in self.transition_rewards_collected:
                    components["pass_success_reward"][rew_index] = self.pass_success
                    self.transition_rewards_collected[transition_key] = True

            # Accumulate additional reward components
            reward[rew_index] += components["pass_success_reward"][rew_index] + components["continuous_control_reward"][rew_index]

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
        return observation, reward, done, info
