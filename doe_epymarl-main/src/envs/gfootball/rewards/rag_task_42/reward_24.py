import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages strategic midfield control and transitions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_control_rewards = {}
        self.control_weight = 1.5
        self.transition_weight = 2.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.midfield_control_rewards = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_control_rewards'] = self.midfield_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control_rewards = from_pickle['midfield_control_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control": [0.0] * len(reward),
                      "strategic_transition": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Control the midfield: We give reward for maintaining position in the midfield,
            # and transitioning roles and positions strategically under pressure.
            ball_pos = o['ball'][0]

            # Identify midfield region and control
            is_midfield = -0.2 <= ball_pos <= 0.2
            
            # Reward midfield control when the team owns the ball
            if is_midfield and o['ball_owned_team'] == 0:  # Assuming team 0 is ours
                if rew_index not in self.midfield_control_rewards:
                    components['midfield_control'][rew_index] = self.control_weight
                    reward[rew_index] += components['midfield_control'][rew_index]
                    self.midfield_control_rewards[rew_index] = True

            # Reward strategic transitions: changing formation or positions when losing ball control
            if o['game_mode'] in [2, 3, 4, 5]:  # These modes represent set pieces where transitions are crucial
                components['strategic_transition'][rew_index] = self.transition_weight
                reward[rew_index] += components['strategic_transition'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
