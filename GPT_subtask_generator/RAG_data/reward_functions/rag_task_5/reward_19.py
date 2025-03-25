import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds defensive skills training reinforcement based on tactical responses and quick transition for counter-attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_recovery_reward = 0.1
        self._counter_attack_effectiveness = 0.2
        self._defensive_transition_checkpoints = 5
        self.ball_possession_tracker = {}
        self.previous_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_possession_tracker = {}
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'ball_possession_tracker': self.ball_possession_tracker,
            'previous_ball_position': self.previous_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_possession_tracker = from_pickle['CheckpointRewardWrapper']['ball_possession_tracker']
        self.previous_ball_position = from_pickle['CheckpointRewardWrapper']['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_recovery_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if 'ball_owned_team' in o:
                if o['ball_owned_team'] == 1 - o['active']:
                    # Potential ball recovery
                    if o['ball'] != self.previous_ball_position:
                        components["defensive_recovery_reward"][rew_index] = self._ball_recovery_reward
                        self.previous_ball_position = o['ball']
                    reward[rew_index] += components["defensive_recovery_reward"][rew_index]
                
                # Implementing reward for counter attacks
                if o['ball_owned_team'] == 1 and self.ball_possession_tracker.get('owner_team', None) != o['ball_owned_team']:
                    distance_moved_towards_opp_goal = self.previous_ball_position[0] - o['ball'][0]
                    if distance_moved_towards_opp_goal > 0:
                        # Encourage forward movement during counter-attack
                        components["counter_attack_reward"][rew_index] = self._counter_attack_effectiveness * distance_moved_towards_opp_goal
                        reward[rew_index] += components["counter_attack_reward"][rew_index]
            
            # Update ball possession tracker
            self.ball_possession_tracker['owner_team'] = o['ball_owned_team']
            self.ball_possession_tracker['owner_index'] = o['ball_owned_player']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
