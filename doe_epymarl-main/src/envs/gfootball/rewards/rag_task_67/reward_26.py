import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages skills like Short Pass, Long Pass, and Dribble in transition from defense to attack."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_control_rewards = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.ball_control_rewards = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_control_rewards'] = self.ball_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_control_rewards = from_pickle.get('ball_control_rewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for index, obs in enumerate(observation):
            ball_pos = obs['ball'][:2]
            team_ball_control = obs['ball_owned_team']
            my_position = obs['right_team' if team_ball_control == 1 else 'left_team']
            opponents = obs['left_team' if team_ball_control == 1 else 'right_team']
            
            if team_ball_control == 1 or team_ball_control == 0:  # If either team controls the ball
                dist_to_goal = np.abs(ball_pos[0] - (1 if team_ball_control == 1 else -1))
                opponent_distances = np.linalg.norm(my_position - opponents, axis=1)
                pass_successful = min(opponent_distances) > 0.3  # No opponent is really close
                dribbling = np.sum(obs['sticky_actions'][8:10]) > 0  # Dribble or sprint actions

                transition_reward = 0
                if team_ball_control == 1 and not dribbling:
                    transition_reward += 0.01 * (1 - dist_to_goal)  # Reward for having the ball closer to opponent's goal
                if pass_successful:
                    transition_reward += 0.05  # Reward for maintaining possession while transitioning

                # Record component rewards
                components.setdefault('transition_reward', []).append(transition_reward)

                # Update the reward
                reward[index] += transition_reward
            else:
                components.setdefault('transition_reward', []).append(0)

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
