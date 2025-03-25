import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards based on midfield and defensive transitions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define midfield transition points
        self.midfield_transition_zones = [0.2, 0.4, 0.6, 0.8]  # Progressive zones in the x-axis of the field
        # Define rewards for successful transitions and actions
        self.transition_reward = 0.05
        self.pass_reward = 0.1
        self.dribble_reward = 0.1
        self.sprint_reward = 0.05
        self.positive_zones_collected = set()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positive_zones_collected.clear()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['positive_zones_collected'] = self.positive_zones_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.positive_zones_collected = from_pickle['positive_zones_collected']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "transition_reward": [0.0] * len(reward),
            "action_rewards": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_x_pos = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]

            # Reward moving through midfield transitions
            for zone in self.midfield_transition_zones:
                if player_x_pos > zone and zone not in self.positive_zones_collected:
                    components["transition_reward"][rew_index] = self.transition_reward
                    reward[rew_index] += components["transition_reward"][rew_index]
                    self.positive_zones_collected.add(zone)

            # Reward specific actions: High Pass, Long Pass, Dribble, and Sprint
            if o['sticky_actions'][8] == 1:  # Sprint
                components["action_rewards"][rew_index] += self.sprint_reward
            if o['sticky_actions'][9] == 1:  # Dribble
                components["action_rewards"][rew_index] += self.dribble_reward
            # Assuming action indices for High Pass and Long Pass are known (example indices 2 and 3)
            if o['sticky_actions'][2] == 1 or o['sticky_actions'][3] == 1:  # High Pass or Long Pass
                components["action_rewards"][rew_index] += self.pass_reward

            reward[rew_index] += components["action_rewards"][rew_index]

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
