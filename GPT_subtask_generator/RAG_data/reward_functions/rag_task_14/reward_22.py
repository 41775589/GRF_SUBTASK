import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focusing on the 'sweeper' defensive role in football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearances = {}
        self.tackles = {}
        self.recoveries = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearances = {}
        self.tackles = {}
        self.recoveries = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'clearances': self.clearances,
            'tackles': self.tackles,
            'recoveries': self.recoveries
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.clearances = state_data['clearances']
        self.tackles = state_data['tackles']
        self.recoveries = state_data['recoveries']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "clearance_reward": [0.0] * len(reward),
            "tackle_reward": [0.0] * len(reward),
            "recovery_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Assign rewards for defensive actions related to the 'sweeper' role
        for rew_index, o in enumerate(observation):
            player_role = o['left_team_roles'][o['active']] if 'left_team_roles' in o else None
            # Reward for successful clearances based on ball out of danger zone
            if player_role == 2 and o['ball'][0] < -0.8:  # Arbitrary value indicating "close to own goal"
                self.clearances[rew_index] = self.clearances.get(rew_index, 0) + 1
                components['clearance_reward'][rew_index] += 0.3  # Define a reward value for clearances

            # Reward for tackles especially last-man tackles
            if player_role == 2 and self.sticky_actions_counter[rew_index] == 0:
                self.tackles[rew_index] = self.tackles.get(rew_index, 0) + 1
                components['tackle_reward'][rew_index] += 0.2
                
            # Reward for quick recoveries to defensive positions
            if player_role == 2 and o['ball_owned_team'] != 0:
                self.recoveries[rew_index] = self.recoveries.get(rew_index, 0) + 1
                components['recovery_reward'][rew_index] += 0.1
            
            # Aggregate all reward components
            reward[rew_index] = components['base_score_reward'][rew_index] \
                + components['clearance_reward'][rew_index] \
                + components['tackle_reward'][rew_index] \
                + components['recovery_reward'][rew_index]

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
