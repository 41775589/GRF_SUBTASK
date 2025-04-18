import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for energy conservation strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_steps = 10  # Number of steps to focus on the efficient move-sprint cycles
        self.move_sprint_efficiency_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and reward tracking variables"""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment along with reward specific data"""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the environment state and retrieve reward specific variables"""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """Modify the rewards by adding penalties for inefficient movement and sprint management"""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "conservation_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Agent aligns its "action_stop" use vs "action_sprint" stimulus
            # Promote not sprinting or moving all the time
            stop_action = o['sticky_actions'][0]  # i.e., stop
            sprint_action = o['sticky_actions'][8]  # i.e., sprint

            # Reward efficient sprint usage: punish sprint use if moving too little
            if not stop_action and sprint_action:
                components["conservation_reward"][rew_index] = self.move_sprint_efficiency_reward
                reward[rew_index] += components["conservation_reward"][rew_index]
                
        return reward, components

    def step(self, action):
        """Step through the environment, modify rewards, and collect diagnostics"""
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
