import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward specific to defensive task training in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the reward counter and environment state."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve the state of the reward calculations along with the environment's state."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the reward calculations based on external loading."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Calculate enhanced reward function focused on defensive actions and coherence."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        for i, o in enumerate(observation):
            # Encourage maintaining a defensive line
            if o['ball_owned_team'] == 0:  # if the opposing team owns the ball
                distance = np.linalg.norm(o['ball'] - o['left_team'][o['active']])
                # Reward for being close to the ball defensively (closer the better)
                components.setdefault('defensive_positioning', []).append((0.5 / (distance + 0.1)))
            else:
                components.setdefault('defensive_positioning', []).append(0.0)

            # Stress the importance of maintaining low tiredness levels
            if o['left_team_tired_factor'][o['active']] < 0.1:
                components.setdefault('stamina_management', []).append(0.1)
            else:
                components.setdefault('stamina_management', []).append(-0.1 * o['left_team_tired_factor'][o['active']])

        # Combine the rewards based on strategic emphasis coefficients
        for key, value in components.items():
            reward += np.array(value)
        
        return reward.tolist(), components

    def step(self, action):
        """Step function that includes detailed reward feedback."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Maintaining sticky actions for logging purposes
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
