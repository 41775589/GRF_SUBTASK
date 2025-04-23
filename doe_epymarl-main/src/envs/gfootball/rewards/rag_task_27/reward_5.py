import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective defensive actions and positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define interception checkpoints as well as defensive positioning zones.
        self._num_defensive_zones = 3
        self._intercept_checkpoint_reward = 0.05
        self._defensive_positioning_reward = 0.02
        self._collected_defensive_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_defensive_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_defensive_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_defensive_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Initialize component structure for detailed reward tracking
        components = {
            "base_score_reward": reward.copy(),
            "intercept_checkpoint_reward": [0.0] * len(reward),
            "defensive_positioning_reward": [0.0] * len(reward)
        }
        
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            if self.env.unwrapped.done:
                break

            # Check if it is the active defender and the opponent has the ball
            if o['ball_owned_team'] == 1:  # assuming the agent's team is 0
                opponent_has_ball = True
            else:
                opponent_has_ball = False
                continue  # Skip further calculations if the ball is not owned by opponents.

            # Check interception capability
            ball_distance = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']][:2])
            if ball_distance < 0.1 and opponent_has_ball:
                # Ball is close to the defender, reward interception attempt
                if not self._collected_defensive_rewards.get('intercept', False):
                    components['intercept_checkpoint_reward'][i] = self._intercept_checkpoint_reward
                    reward[i] += components['intercept_checkpoint_reward'][i]
                    self._collected_defensive_rewards['intercept'] = True

            # Check defensive positioning (opponent's half closer to own goal)
            if o['left_team'][o['active']][0] < -0.5:
                if not self._collected_defensive_rewards.get('defensive_positioning', False):
                    components['defensive_positioning_reward'][i] = self._defensive_positioning_reward
                    reward[i] += components['defensive_positioning_reward'][i]
                    self._collected_defensive_rewards['defensive_positioning'] = True

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
