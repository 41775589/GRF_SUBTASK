import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for executing high passes with precision in specific scenarios.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_height_threshold = 0.1  # Define a threshold for considering a pass as 'high'
        self.high_pass_reward = 1.0        # Reward for each high and accurate pass
        self.player_positions_to_monitor = {}  # To keep track of positions when high pass starts

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_positions_to_monitor = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.player_positions_to_monitor
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.player_positions_to_monitor = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0, 0.0]}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if high pass scenario: ball goes above a certain height!
            if o['ball'][2] > self.ball_height_threshold and o['ball_direction'][2] > 0:
                pass_initiator = o['ball_owned_player']
                if pass_initiator is not None and o['ball_owned_team'] == 0:  # Assuming team 0 is our team
                    team = 'left_team' if o['ball_owned_team'] == 0 else 'right_team'

                    # Monitor initiating and receiving positions when the ball is ascending
                    if pass_initiator not in self.player_positions_to_monitor:
                        self.player_positions_to_monitor[pass_initiator] = o[team][pass_initiator]
                    
                    ball_position = np.asarray(o['ball'][:2])
                    for player_idx, player_pos in enumerate(o[team]):
                        distance_to_ball = np.linalg.norm(player_pos - ball_position)
                        
                        # Identify if a high pass is completed to a teammate and not intercepted or out of play
                        if distance_to_ball < 0.1 and player_idx != pass_initiator:
                            components["high_pass_reward"][rew_index] = self.high_pass_reward
                            reward[rew_index] += components["high_pass_reward"][rew_index]
                            self.player_positions_to_monitor.pop(pass_initiator, None)  # Clear after rewarding

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
                if action > 0:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
