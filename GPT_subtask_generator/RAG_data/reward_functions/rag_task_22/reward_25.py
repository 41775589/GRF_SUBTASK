import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense sprint reward focused on improving defensive coverage."""

    def __init__(self, env):
        super().__init__(env)
        self.sprint_distance_threshold = 0.05  # Distance covered to consider rewarding sprint
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "sprint_repositioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i, obs in enumerate(observation):
            # Detect if the sprint action is active
            sprint_action_active = obs['sticky_actions'][8]  # sprint action index 8
            if sprint_action_active:
                self.sticky_actions_counter[i] += 1
            
            # Compute the displacement magnitude of the agent
            position = obs.get('left_team' if obs['ball_owned_team'] == 0 else 'right_team', None)
            if position is not None:
                player_idx = obs['active']
                current_position = position[player_idx]
                previous_position = position[player_idx - 1] if player_idx > 0 else current_position
                displacement = np.linalg.norm(previous_position - current_position)

                # Reward sprinting if it leads to significant repositioning
                if displacement > self.sprint_distance_threshold:
                    components['sprint_repositioning_reward'][i] += displacement * 0.1

        # Sum the rewards
        for i in range(len(reward)):
            reward[i] += components['sprint_repositioning_reward'][i]
          
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
