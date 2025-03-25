import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance the synergistic effectiveness of the central midfield by focusing on transitions and pace management."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_checkpoints = {}
        self.num_transition_zones = 5
        self.transition_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.transition_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.transition_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Central midfielders control id definitions
            if o['active'] not in [4, 5, 6]:  # assuming indices 4, 5, 6 are central midfielders
                continue

            # Check for controlled pace, using player speed and tiredness
            player_speed = np.linalg.norm(o['left_team_direction'][o['active']])
            if player_speed > 0.05 and not o['left_team_tired_factor'][o['active']] > 0.1:
                # Dividing the field into 5 zones and giving reward for transitions
                current_zone = int((o['left_team'][o['active']][0] + 1) / 0.4)  # Mapping -1,1 to 0-5
                if current_zone in range(1, 5):
                    if current_zone not in self.transition_checkpoints:
                        components["transition_reward"][rew_index] = self.transition_reward
                        reward[rew_index] += components["transition_reward"][rew_index]
                        self.transition_checkpoints[current_zone] = True

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
