import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward specifically focused on the timing and precision of sliding tackles under high-pressure situations in a defensive football setup."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sliding_tackle_success_reward = 1.0
        self.ball_interception_reward = 0.5
        self.defensive_positioning_reward = 0.2
        self.penalty_for_foul = -0.5

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sliding_tackle_success": 0.0,
                      "ball_interception": 0.0,
                      "defensive_positioning": 0.0,
                      "fouls_committed": 0.0}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            active_player_position = o['left_team'][o['active']]
            ball_position = o['ball'][:2]
            distance_to_ball = np.linalg.norm(ball_position - active_player_position)

            # Reward successful sliding tackles
            if o['game_mode'] == 6 and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components['sliding_tackle_success'] += self.sliding_tackle_success_reward

            # Reward interception of the ball not necessarily via slide but timely possession gain
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components['ball_interception'] += self.ball_interception_reward

            # Reward for maintaining good defensive positioning
            if distance_to_ball < 0.15:  # Arbitrary threshold for 'good' distance
                components['defensive_positioning'] += self.defensive_positioning_reward

            # Penalty for committing fouls (if game mode indicates foul by this player)
            if o['game_mode'] in [2, 3, 4] and o['ball_owned_team'] != 0:
                components['fouls_committed'] += self.penalty_for_foul

            reward[rew_index] += sum(components.values())

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Add each individual reward component to the info
        for key in components:
            info[f"component_{key}"] = components[key]

        return observation, reward, done, info
