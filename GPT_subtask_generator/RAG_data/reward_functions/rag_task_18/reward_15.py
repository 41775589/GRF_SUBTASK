import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards precise control and synergy in midfield transitions.
    Focus on ball control, passing effectiveness in the central areas, and maintaining tactical formations."""

    def __init__(self, env):
        super().__init__(env)
        self.control_threshold = 0.1  # Define a threshold to consider effective ball control
        self.midfield_rewards = np.linspace(-0.5, 0.5, num=5)  # Central midfield sections
        self.pass_reward = 0.2
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
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward),
                      "pass_effectiveness_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            controlled_player = o["active"]
            # Determine if controlled player is in the central midfield and if the ball is effectively controlled
            if o["ball_owned_team"] == 0 and o["ball_owned_player"] == controlled_player:
                player_pos = o["left_team"][controlled_player]
            elif o["ball_owned_team"] == 1 and o["ball_owned_player"] == controlled_player:
                player_pos = o["right_team"][controlled_player]
            else:
                continue
            
            # Check if player position is within midfield boundaries and adjust rewards
            if -0.2 <= player_pos[0] <= 0.2:
                reward_index = int((player_pos[1] + 0.5) / 0.2)
                components["midfield_control_reward"][rew_index] = self.midfield_rewards[reward_index]
                reward[rew_index] += components["midfield_control_reward"][rew_index]

            # Reward for successful passes
            if 'sticky_actions' in o and o['sticky_actions'][6]:
                components["pass_effectiveness_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_effectiveness_reward"][rew_index]

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
