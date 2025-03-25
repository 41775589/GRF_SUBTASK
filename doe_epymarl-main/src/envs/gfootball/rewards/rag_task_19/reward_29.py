import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on defense and strategic midfield management."""

    def __init__(self, env):
        super().__init__(env)
        self.defense_positions = np.linspace(-1.0, 0, 5)  # From the left goal to the midfield
        self.midfield_positions = np.linspace(0, 1.0, 5)  # From  midfield to the right goal
        self.defense_collected = False
        self.midfield_collected = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.defense_collected = False
        self.midfield_collected = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_data = {'defense_collected': self.defense_collected,
                      'midfield_collected': self.midfield_collected}
        to_pickle['CheckpointRewardWrapper'] = state_data
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.defense_collected = state_data['defense_collected']
        self.midfield_collected = state_data['midfield_collected']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_management_reward": [0.0] * len(reward),
                      "midfield_management_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Defensive reward: Award when any team player is near the defensive strategic positions
            if not self.defense_collected and o['ball_owned_team'] == 0:
                if any(np.isclose(o['left_team'][:, 0], self.defense_positions, atol=0.1).any()):
                    components["defense_management_reward"][idx] = 0.2
                    reward[idx] += components["defense_management_reward"][idx]
                    self.defense_collected = True
            
            # Midfield reward: Award when any team player controls the ball at strategic midfield positions
            if not self.midfield_collected and o['ball_owned_team'] == 0 and \
               o['ball_owned_player'] == o['active']:
                if any(np.isclose(o['left_team'][:, 0], self.midfield_positions, atol=0.1).any()):
                    components["midfield_management_reward"][idx] = 0.3
                    reward[idx] += components["midfield_management_reward"][idx]
                    self.midfield_collected = True

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
