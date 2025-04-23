import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for strategic defensive transitioning behaviour,
       focusing on lateral movements and quick repositioning to a defensive stance."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.held_positions = {}
        self.defensive_transitions_counter = 0
        self.transition_reward = 1.0
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.held_positions = {}
        self.defensive_transitions_counter = 0
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['held_positions'] = self.held_positions
        to_pickle['defensive_transitions_counter'] = self.defensive_transitions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.held_positions = from_pickle['held_positions']
        self.defensive_transitions_counter = from_pickle['defensive_transitions_counter']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for index in range(len(reward)):
            curr_obs = observation[index]
            player_x, player_y = curr_obs['right_team'][curr_obs['active']][:2]

            # Track defensive positions held
            if self.held_positions.get(index) is None:
                self.held_positions[index] = []

            if len(self.held_positions[index]) == 0 or self._position_diff(self.held_positions[index][-1], (player_x, player_y)) > 0.1:
                self.held_positions[index].append((player_x, player_y))
                if len(self.held_positions[index]) > 1:
                    if self._is_defensive_transition(self.held_positions[index][-2], self.held_positions[index][-1]):
                        reward[index] += self.transition_reward
                        components["transition_reward"][index] = self.transition_reward
                        self.defensive_transitions_counter += 1
                        # Reward for transitioning back to defensive position quickly

        return reward, components

    def _position_diff(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _is_defensive_transition(self, previous, current):
        # Condition for lateral movement or quick repositioning
        return abs(previous[0] - current[0]) > 0.5 and abs(previous[1] - current[1]) < 0.1
    
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
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
