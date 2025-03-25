import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on offensive play and rapid advancement."""

    def __init__(self, env):
        super().__init__(env)
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.2
        self._collected_checkpoints = {}
        self._possession_change_penalty = -0.1
        self._score_multiplier = 2.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self._collected_checkpoints = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward),
                      "possession_change_penalty": [0.0] * len(reward),
                      "score_multiplier_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if o['game_mode'] != 0:  # Only calculate in normal play mode
                continue
            
            # Manage possession change penalty
            if o['ball_owned_team'] != self._collected_checkpoints.get(rew_index, {}).get('last_possession', None):
                components["possession_change_penalty"][rew_index] = self._possession_change_penalty
                self._collected_checkpoints[rew_index]['last_possession'] = o['ball_owned_team']

            # Calculate progress reward towards the opponent's goal
            if o['ball_owned_team'] == 1:  # Currently assumes environment is from the perspective of the right team
                checkpoints_collected = self._collected_checkpoints.setdefault(rew_index, {}).setdefault('checkpoints', 0)
                ball_y_position = o['ball'][1]  # Only considering Y-axis movement towards goal
                    
                if ball_y_position > checkpoints_collected * 0.2:
                    components["checkpoint_reward"][rew_index] += self._checkpoint_reward
                    self._collected_checkpoints[rew_index]['checkpoints'] += 1

            # Score multiplier
            if o['score'][0] > 0 or o['score'][1] > 0:  # some team scored
                components["score_multiplier_bonus"][rew_index] = self._score_multiplier * (o['score'][0] + o['score'][1])
                
            # aggregates all reward components
            reward[rew_index] = sum([
                components["base_score_reward"][rew_index],
                components["checkpoint_reward"][rew_index],
                components["possession_change_penalty"][rew_index],
                components["score_multiplier_bonus"][rew_index]
            ])
            
        return reward, components

    def step(self, action):
        # This part should not change according to the prompt.
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
