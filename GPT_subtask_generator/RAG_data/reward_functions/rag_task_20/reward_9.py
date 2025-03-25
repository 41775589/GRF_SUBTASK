import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on offensive strategies
    in football, optimizing team coordination and attacking skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards_collected = {}
        self.pass_rewards_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards_collected = {}
        self.pass_rewards_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['position_rewards'] = self.position_rewards_collected
        state['pass_rewards'] = self.pass_rewards_collected
        return state

    def set_state(self, state):
        res = self.env.set_state(state)
        self.position_rewards_collected = state['position_rewards']
        self.pass_rewards_collected = state['pass_rewards']
        return res

    def reward(self, reward):
        # Calculate modified rewards based on strategic positions and good passes
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward for maintaining optimal positions for potential attacks
            x_position = o['right_team'][o['active']][0]  # X-position of controlled player

            if x_position > 0.5:
                key_position = f"position_{i}"
                if key_position not in self.position_rewards_collected:
                    self.position_rewards_collected[key_position] = True
                    components["positioning_reward"][i] = 0.2  # position in the opponent's half
            
            # Reward for successful passes
            if o['game_mode'] == 1:  # 1 = Game mode for successful passes
                key_pass = f"pass_{i}"
                if key_pass not in self.pass_rewards_collected:
                    self.pass_rewards_collected[key_pass] = True
                    components["passing_reward"][i] = 0.5
        
        # Update rewards with components for each agent
        for j in range(len(reward)):
            reward[j] += components["positioning_reward"][j] + components["passing_reward"][j]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
