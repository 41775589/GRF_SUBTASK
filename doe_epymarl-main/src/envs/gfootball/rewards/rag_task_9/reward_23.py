import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a customized reward for developing offensive skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define reward parameters
        self._pass_reward = 0.05
        self._shot_reward = 0.1
        self._dribble_reward = 0.02
        self._sprint_reward = 0.01
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, rewards):
        """Adjust rewards based on the offensive actions performed."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(rewards, dtype=float)}
        
        if observation is None:
            return rewards, components

        components['pass_reward'] = np.zeros(len(rewards))
        components['shot_reward'] = np.zeros(len(rewards))
        components['dribble_reward'] = np.zeros(len(rewards))
        components['sprint_reward'] = np.zeros(len(rewards))
        
        for i, o in enumerate(observation):
            # Rewards for passing
            if 'action_pass' in o.get('sticky_actions', []):
                components['pass_reward'][i] = self._pass_reward
            # Rewards for shooting
            if 'action_shot' in o.get('sticky_actions', []):
                components['shot_reward'][i] = self._shot_reward
            # Rewards for dribbling
            if 'action_dribble' in o.get('sticky_actions', []):
                components['dribble_reward'][i] = self._dribble_reward
            # Rewards for sprinting
            if 'action_sprint' in o.get('sticky_actions', []):
                components['sprint_reward'][i] = self._sprint_reward

            # Aggregate customized rewards into the main reward array
            total_custom_reward = (components['pass_reward'][i] +
                                   components['shot_reward'][i] +
                                   components['dribble_reward'][i] +
                                   components['sprint_reward'][i])
            rewards[i] += total_custom_reward

        return rewards, components

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
