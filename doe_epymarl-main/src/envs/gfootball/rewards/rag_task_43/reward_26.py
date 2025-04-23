import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive reward based on player positions and ball ownership."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones = np.linspace(-1, 1, 10)  # Dividing the field into 10 zones longitudinally
        self.zone_reward = 0.02
        self.counterattack_bonus = 0.5
        self.last_ball_pos = 0.0
        self.ball_ownership_changes = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_pos = 0.0
        self.ball_ownership_changes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_zones'] = self.defensive_zones
        to_pickle['last_ball_pos'] = self.last_ball_pos
        to_pickle['ball_ownership_changes'] = self.ball_ownership_changes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_zones = from_pickle['defensive_zones']
        self.last_ball_pos = from_pickle['last_ball_pos']
        self.ball_ownership_changes = from_pickle['ball_ownership_changes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_position_reward": [0.0] * len(reward),
                      "counterattack_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        current_ball_pos = observation[0]['ball'][0]  # accessing ball's x position
        ball_owned_team = observation[0]['ball_owned_team']
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_x_pos = o['left_team'][rew_index][0] if o['left_team_active'][rew_index] else o['right_team'][rew_index][0]

            # Determine the zone of the player and grant rewards for being in defensive zones based on ball position
            for i, zone_edge in enumerate(self.defensive_zones):
                if player_x_pos < zone_edge:
                    components["defensive_position_reward"][rew_index] = self.zone_reward * (10 - i)
                    break

            # Reward for quick transitions from defense to attack (counterattacks)
            if self.last_ball_pos is not None:
                if ball_owned_team == 1 and self.last_ball_pos < current_ball_pos and o['ball_owned_team'] == 0:
                    components["counterattack_reward"][rew_index] = self.counterattack_bonus
        
        # Update info for next step
        self.last_ball_pos = current_ball_pos

        # Aggregate rewards
        for rew_index in range(len(reward)):
            reward[rew_index] += components["defensive_position_reward"][rew_index] + components["counterattack_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
