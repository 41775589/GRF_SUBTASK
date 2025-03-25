import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper enhancing the skills of a 'stopper' in football - focusing on blocking and interceptions."""

    def __init__(self, env):
        super().__init__(env)
        self.reset()

    def reset(self):
        """Resets the accumulators and other stateful objects."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Records the current state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Custom reward function to enhance stopper skills."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "block_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]

            # Incremental reward for intercepting the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if o['game_mode'] in [3, 5]:  # free-kick or throw-in, i.e., regaining control
                    components["interception_reward"][idx] += 1.0
                    reward[idx] += components["interception_reward"][idx]
            
            # Incremental reward for blocking the ball
            if o['ball_owned_team'] == 1:  # when opposing team has the ball
                distance_to_ball = np.linalg.norm(o['left_team'][o['active']] - o['ball'][:2])
                if distance_to_ball < 0.1:  # very close to the ball
                    components["block_reward"][idx] += 0.5
                    reward[idx] += components["block_reward"][idx]

        return reward, components

    def step(self, action):
        """Step function to execute actions and modify rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
