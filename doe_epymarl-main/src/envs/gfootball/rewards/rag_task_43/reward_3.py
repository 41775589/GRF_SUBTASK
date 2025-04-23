import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive and counter-attack strategy reward based on positional awareness and responsiveness."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_opponent_distance = np.inf
        self.ball_position_last_step = None
        self.reward_for_ball_recovery = 1.0
        self.reward_for_effective_clearance = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_opponent_distance = np.inf
        self.ball_position_last_step = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['last_opponent_distance'] = self.last_opponent_distance
        to_pickle['ball_position_last_step'] = self.ball_position_last_step
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.last_opponent_distance = from_pickle['last_opponent_distance']
        self.ball_position_last_step = from_pickle['ball_position_last_step']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "clearance_reward": [0.0] * len(reward), "recovery_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][:2]  # Get ball position excluding z-coordinate
            ball_owned_team = o['ball_owned_team']
            defense_team = 0 if ball_owned_team == 1 else 1  # Assume agent team to be 0

            # Calculate distance to closest opponent
            team_oppo = 'right_team' if defense_team == 0 else 'left_team'
            opponent_distances = [np.linalg.norm(np.array(player_pos) - ball_pos) for player_pos in o[team_oppo]]
            min_opponent_distance = min(opponent_distances)
            
            # Check for successful ball recovery
            if self.ball_position_last_step is not None and ball_owned_team == defense_team:
                dist_moved = np.linalg.norm(np.array(ball_pos) - np.array(self.ball_position_last_step))
                if dist_moved > 0.1 and min_opponent_distance < self.last_opponent_distance:
                    components['recovery_reward'][rew_index] = self.reward_for_ball_recovery
                    reward[rew_index] += components['recovery_reward'][rew_index]
            
            # Reward clearance if the ball has been sent significantly towards the opponent side
            if ball_owned_team == defense_team and ball_pos[0] * (1 if defense_team == 0 else -1) > 0.5:
                components['clearance_reward'][rew_index] = self.reward_for_effective_clearance
                reward[rew_index] += components['clearance_reward'][rew_index]

            # Update states for next use
            self.last_opponent_distance = min_opponent_distance
            self.ball_position_last_step = ball_pos

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
