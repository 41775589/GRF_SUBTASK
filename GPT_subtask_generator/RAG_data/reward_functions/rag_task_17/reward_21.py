import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a wide midfield mastery and passing reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._pass_checkpoint_achieved = {}
        self._wide_position_checkpoint_achieved = {}
        self._passing_reward = 0.3
        self._wide_position_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pass_checkpoint_achieved = {}
        self._wide_position_checkpoint_achieved = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_checkpoint'] = self._pass_checkpoint_achieved
        to_pickle['wide_position_checkpoint'] = self._wide_position_checkpoint_achieved
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._pass_checkpoint_achieved = from_pickle.get('pass_checkpoint', {})
        self._wide_position_checkpoint_achieved = from_pickle.get('wide_position_checkpoint', {})
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "wide_position_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                # Assess successful pass effectiveness
                if 'action' in o and o['action'] == 'high_pass':
                    if rew_index not in self._pass_checkpoint_achieved:
                        components['passing_reward'][rew_index] += self._passing_reward
                        self._pass_checkpoint_achieved[rew_index] = True

                # Assess wide midfield positioning
                if any(player_pos[0] > 0.75 for player_pos in o['left_team']):
                    if rew_index not in self._wide_position_checkpoint_achieved:
                        components['wide_position_reward'][rew_index] += self._wide_position_reward
                        self._wide_position_checkpoint_achieved[rew_index] = True
            
            reward[rew_index] += components['passing_reward'][rew_index] + components['wide_position_reward'][rew_index]

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) # Sum up individual components for logging
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
