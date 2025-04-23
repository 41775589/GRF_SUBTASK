import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds scenario-based rewards focusing on passing and shooting in football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Setup strategic checkpoints in the field for passing and shooting practice
        self.pass_checkpoints = [(-0.5, 0), (0, 0.3), (0, -0.3), (0.5, 0)]
        self.shoot_checkpoints = [(0.8, -0.2), (0.8, 0.2), (1, 0)]
        self.checkpoint_radius = 0.1
        self.checkpoint_reward = 0.05
        self.goal_reward = 1  # Big reward for scoring a goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for index in range(len(reward)):
            o = observation[index]
            ball_pos = o['ball']
            shooting = False

            # Check pass checkpoints
            for checkpoint in self.pass_checkpoints:
                if np.linalg.norm(np.subtract(ball_pos[:2], checkpoint)) < self.checkpoint_radius:
                    reward[index] += self.checkpoint_reward
                    components['passing_reward'][index] += self.checkpoint_reward
            
            # Check shoot checkpoints
            for checkpoint in self.shoot_checkpoints:
                if np.linalg.norm(np.subtract(ball_pos[:2], checkpoint)) < self.checkpoint_radius:
                    reward[index] += self.checkpoint_reward
                    components['shooting_reward'][index] += self.checkpoint_reward
                    shooting = True

            # Additional reward for scoring
            if o['score'][1] > o['score'][0]:  # assuming the agent is on team right
                reward[index] += self.goal_reward
                if shooting:
                    components['shooting_reward'][index] += self.goal_reward

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
