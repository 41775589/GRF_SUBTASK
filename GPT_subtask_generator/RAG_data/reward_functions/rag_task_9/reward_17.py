import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on offensive football skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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
        """Append extra reward components engaging in passes, shots, sprints, dribbles."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for controlled passes
            if 'short_pass' in o['sticky_actions'] and o['sticky_actions']['short_pass']:
                components["passing_reward"][rew_index] = 0.05
            if 'long_pass' in o['sticky_actions'] and o['sticky_actions']['long_pass']:
                components["passing_reward"][rew_index] = 0.05
            
            # Reward for attempting shots
            if 'shot' in o['sticky_actions'] and o['sticky_actions']['shot']:
                components["shooting_reward"][rew_index] = 0.1

            # Reward for sprinting effectively
            if 'sprint' in o['sticky_actions'] and o['sticky_actions']['sprint']:
                if o["ball_owned_team"] == 0 and o["ball_owned_player"] == o['active']:
                    components["sprint_reward"][rew_index] = 0.02

            # Reward for dribbling
            if 'dribble' in o['sticky_actions'] and o['sticky_actions']['dribble']:
                if o["ball_owned_team"] == 0 and o["ball_owned_player"] == o['active']:
                    components["dribble_reward"][rew_index] = 0.03

            # Summing individual components to form the final reward for current index
            reward[rew_index] += sum(components[comp][rew_index] for comp in components)

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
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
