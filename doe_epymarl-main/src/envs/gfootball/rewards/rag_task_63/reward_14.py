import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to train a goalkeeper on various skills including shot stopping, 
    ball distribution decision-making under pressure, and communication with defenders."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize goalkeeper-specific metrics
        self._player_roles = {0: 'GoalKeeper', 1: 'Defender'}
        self._communication_reward = 0.1
        self._distribution_reward = 0.2
        self._shot_stopping_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "communication_reward": [0.0] * len(reward),
            "distribution_reward": [0.0] * len(reward),
            "shot_stopping_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Goalkeeper specific rewards
            if self._player_roles.get(o['active'], None) == 'GoalKeeper':
                # Shot stopping: reward goalkeeper for saves when the ball is close and moving quickly toward the goal
                if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball_direction']) > 0.5:
                    distance_to_goal = abs(o['ball'][0] - 1)  # Assuming goal at x = 1
                    if distance_to_goal < 0.2:
                        components['shot_stopping_reward'][rew_index] = self._shot_stopping_reward
                        reward[rew_index] += components['shot_stopping_reward'][rew_index]

                # Communication with defenders: Reward for maintaining position in relation to defenders
                defenders_positions = [t for r, t in zip(o['right_team_roles'], o['right_team']) if self._player_roles.get(r) == 'Defender']
                if defenders_positions:
                    avg_defender_x = np.mean([pos[0] for pos in defenders_positions])
                    # Reward keeping alignment with defenders
                    if abs(o['right_team'][o['active']][0] - avg_defender_x) < 0.1:
                        components['communication_reward'][rew_index] = self._communication_reward
                        reward[rew_index] += components['communication_reward'][rew_index]

                # Quick decision-making under pressure: Reward for quick releases of ball under opponent pressure
                if o['ball_owned_team'] == 0:
                    opponent_distances = [np.linalg.norm(o['left_team'][i] - o['right_team'][o['active']]) for i in range(len(o['left_team']))]
                    if min(opponent_distances) < 0.3:  # Example threshold for "under pressure"
                        components['distribution_reward'][rew_index] = self._distribution_reward
                        reward[rew_index] += components['distribution_reward'][rew_index]

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
