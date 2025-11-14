import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering Shooting and Dribbling skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_rewards_given = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.shooting_rewards_given = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'shooting_rewards_given': self.shooting_rewards_given}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shooting_rewards_given = from_pickle['CheckpointRewardWrapper']['shooting_rewards_given']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}

        components = {
            "base_score_reward": reward.copy(),
            "shooting_skill_improvement": 0.0
        }

        # Rewarding shooting skills if within shooting range and not already rewarded
        if not self.shooting_rewards_given:
            if 'ball_owned_team' in observation and observation['ball_owned_team'] == 0:  # Assuming 0 is the agent's team
                shooting_distance = np.linalg.norm(observation['ball'][:2])  # 2D distance to (0,0), the goal
                if shooting_distance < 0.2:  # Close enough to be considered a shooting range
                    reward += 2  # Significant reward for getting into shooting position
                    components['shooting_skill_improvement'] = 2
                    self.shooting_rewards_given = True

        # Reward for dribbling: check movement with ball possession
        if 'ball_owned_team' in observation and observation['ball_owned_team'] == 0:  # Team 0 owns the ball
            if 'active' in observation and observation['sticky_actions'][9] == 1:  # Assuming 9 corresponds with dribbling action
                reward += 0.1  # Each step dribbling contributes a smaller reward
                components['shooting_skill_improvement'] += 0.1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
