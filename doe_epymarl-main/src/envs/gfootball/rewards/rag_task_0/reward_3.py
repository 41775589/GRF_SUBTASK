import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for high-speed ball carrying and breakthrough plays."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Ball carrying tracking
        self._last_ball_position = None
        self._last_player_position = None
        self._total_carrying_distance = {}
        self._last_ball_owned = {}
        
        # Defender breakthrough tracking
        self._last_defenders_behind = {}
        self._breakthrough_count = {}
        
        # Penalty area tracking
        self._penalty_area_time = {}
        self._last_in_penalty = {}
        
        # Reward coefficients
        self._carrying_distance_coeff = 0.5
        self._breakthrough_coeff = 0.3
        self._penalty_area_coeff = 0.2
        self._speed_bonus_coeff = 0.1
        
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_ball_position = None
        self._last_player_position = None
        self._total_carrying_distance = {}
        self._last_ball_owned = {}
        self._last_defenders_behind = {}
        self._breakthrough_count = {}
        self._penalty_area_time = {}
        self._last_in_penalty = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'total_carrying_distance': self._total_carrying_distance,
            'last_ball_owned': self._last_ball_owned,
            'last_defenders_behind': self._last_defenders_behind,
            'breakthrough_count': self._breakthrough_count,
            'penalty_area_time': self._penalty_area_time,
            'last_in_penalty': self._last_in_penalty
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_data = from_pickle['CheckpointRewardWrapper']
        self._total_carrying_distance = checkpoint_data['total_carrying_distance']
        self._last_ball_owned = checkpoint_data['last_ball_owned']
        self._last_defenders_behind = checkpoint_data['last_defenders_behind']
        self._breakthrough_count = checkpoint_data['breakthrough_count']
        self._penalty_area_time = checkpoint_data['penalty_area_time']
        self._last_in_penalty = checkpoint_data['last_in_penalty']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "carrying_distance_reward": [0.0] * len(reward),
            "breakthrough_reward": [0.0] * len(reward),
            "penalty_area_reward": [0.0] * len(reward),
            "speed_bonus_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Base score reward (goals)
            if reward[rew_index] == 1:
                reward[rew_index] = 1 * components["base_score_reward"][rew_index]
                continue

            # Check if the active player has the ball
            has_ball = ('ball_owned_team' in o and
                       o['ball_owned_team'] == 0 and
                       'ball_owned_player' in o and
                       o['ball_owned_player'] == o['active'])

            if has_ball:
                active_player_idx = o['active']
                player_pos = o['left_team'][active_player_idx]
                ball_pos = o['ball'][:2]  # x, y coordinates
                
                # Track ball carrying distance (forward progress towards goal)
                if (self._last_ball_owned.get(rew_index, False) and 
                    self._last_player_position is not None):
                    
                    # Calculate forward movement (towards right goal at x=1)
                    forward_distance = player_pos[0] - self._last_player_position[0]
                    if forward_distance > 0:  # Only reward forward movement
                        carrying_reward = forward_distance * self._carrying_distance_coeff
                        components["carrying_distance_reward"][rew_index] = carrying_reward
                        reward[rew_index] += carrying_reward
                        
                        # Speed bonus for sprinting while carrying
                        if len(o['sticky_actions']) > 8 and o['sticky_actions'][8] == 1:  # Sprint action
                            speed_bonus = forward_distance * self._speed_bonus_coeff
                            components["speed_bonus_reward"][rew_index] = speed_bonus
                            reward[rew_index] += speed_bonus

                # Count defenders behind (breakthrough detection)
                right_team = o['right_team']
                current_defenders_behind = 0
                for defender_pos in right_team:
                    if defender_pos[0] < player_pos[0]:  # Defender is behind in x-axis
                        current_defenders_behind += 1
                
                last_defenders = self._last_defenders_behind.get(rew_index, 0)
                if current_defenders_behind > last_defenders:
                    # Successfully moved past defender(s)
                    breakthroughs = current_defenders_behind - last_defenders
                    breakthrough_reward = breakthroughs * self._breakthrough_coeff
                    components["breakthrough_reward"][rew_index] = breakthrough_reward
                    reward[rew_index] += breakthrough_reward
                
                self._last_defenders_behind[rew_index] = current_defenders_behind

                # Penalty area reward (right penalty area: x > 0.83, |y| < 0.21)
                in_penalty_area = (player_pos[0] > 0.83 and abs(player_pos[1]) < 0.21)
                if in_penalty_area:
                    penalty_reward = self._penalty_area_coeff
                    components["penalty_area_reward"][rew_index] = penalty_reward
                    reward[rew_index] += penalty_reward
                
                self._last_in_penalty[rew_index] = in_penalty_area
                self._last_player_position = player_pos.copy()
                self._last_ball_owned[rew_index] = True
                
            else:
                # Player doesn't have ball
                self._last_ball_owned[rew_index] = False
                self._last_player_position = None

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
