import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specifically rewards dribbling and dynamic positioning for
    transitions between defense and offense."""

    def __init__(self, env):
        super().__init__(env)
        # Track the number of dribble actions and their effectiveness
        self.dribble_count = np.zeros(2, dtype=int)
        self.positioning_score = np.zeros(2, dtype=float)
        self.initial_positions = None
        self.previous_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.dribble_count.fill(0)
        self.positioning_score.fill(0.0)
        self.initial_positions = None
        self.previous_ball_position = None
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'dribble_count': self.dribble_count.copy(),
            'positioning_score': self.positioning_score.copy()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        loaded = from_pickle['CheckpointRewardWrapper']
        self.dribble_count = loaded['dribble_count']
        self.positioning_score = loaded['positioning_score']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            'base_score_reward': reward.copy(),
            'dribble_reward': [0.0] * len(reward),
            'positioning_reward': [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        ball_position = np.array(observation['ball'][:2])
        if self.previous_ball_position is None:
            self.previous_ball_position = ball_position
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = np.array(o['left_team' if o['active'] < 11 else 'right_team'][o['active'] % 11])

            # Initialize initial player positions at the start
            if self.initial_positions is None:
                self.initial_positions = {}
                for team in ['left_team', 'right_team']:
                    for i, pos in enumerate(o[team]):
                        self.initial_positions[(team, i)] = np.array(pos)

            initial_pos = self.initial_positions[('left_team' if o['active'] < 11 else 'right_team', o['active'] % 11)]

            # Reward dribbling actions
            if o['sticky_actions'][9] == 1:  # action_dribble
                self.dribble_count[rew_index] += 1
                # Smaller reward incrementally for dribbling longer
                components['dribble_reward'][rew_index] = 0.1 / (1 + self.dribble_count[rew_index])

            # Positioning reward encouraging movement towards ball and fluid transitions
            distance_to_ball_initial = np.linalg.norm(initial_pos - self.previous_ball_position)
            distance_to_ball_now = np.linalg.norm(active_player_pos - ball_position)

            if distance_to_ball_now < distance_to_ball_initial:
                # Reward movement towards the ball
                components['positioning_reward'][rew_index] = 0.5 * (distance_to_ball_initial - distance_to_ball_now)

            # Calculate final reward with components
            reward[rew_index] += components['dribble_reward'][rew_index] + components['positioning_reward'][rew_index]

        self.previous_ball_position = ball_position
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
