import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that introduces rewards for passing, dribbling, and sprint management for a hybrid midfielder/defender agent."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.pass_checkpoints = {}
        self.dribble_checkpoints = {}
        self.sprint_checkpoints = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_checkpoints = {}
        self.dribble_checkpoints = {}
        self.sprint_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_checkpoints'] = self.pass_checkpoints
        to_pickle['dribble_checkpoints'] = self.dribble_checkpoints
        to_pickle['sprint_checkpoints'] = self.sprint_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_checkpoints = from_pickle['pass_checkpoints']
        self.dribble_checkpoints = from_pickle['dribble_checkpoints']
        self.sprint_checkpoints = from_pickle['sprint_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_controlled_player = o['active']
            
            # Checking for pass actions
            if o['sticky_actions'][6]:  # Assuming action 6 represents 'High Pass'
                if self.pass_checkpoints.get(rew_index, 0) < 3:
                    components["pass_reward"][rew_index] += 0.05
                    reward[rew_index] += components["pass_reward"][rew_index]
                    self.pass_checkpoints[rew_index] = self.pass_checkpoints.get(rew_index, 0) + 1

            # Checking for dribble when under pressure
            own_player_pos = o['left_team'][is_controlled_player]
            opponents = o['right_team']
            # Measure pressure by proximity of nearest opponent
            distances = np.sqrt(((opponents - own_player_pos)**2).sum(axis=1))
            if min(distances) < 0.1:  # If close opponent is near
                if o['sticky_actions'][9]:  # Assuming action 9 represents 'Dribble'
                    if self.dribble_checkpoints.get(rew_index, 0) < 5:
                        components["dribble_reward"][rew_index] += 0.1
                        reward[rew_index] += components["dribble_reward"][rew_index]
                        self.dribble_checkpoints[rew_index] = self.dribble_checkpoints.get(rew_index, 0) + 1
            
            # Checking for sprint management
            if o['sticky_actions'][8]:  # Sprint action
                if not self.sprint_checkpoints.get(rew_index):
                    components["sprint_reward"][rew_index] += 0.03
                    reward[rew_index] += components["sprint_reward"][rew_index]
                    self.sprint_checkpoints[rew_index] = True
            else:
                self.sprint_checkpoints[rew_index] = False

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
