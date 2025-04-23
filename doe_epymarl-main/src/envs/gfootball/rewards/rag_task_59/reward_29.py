import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for goalkeeper coordination and ball clearing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        # Enhance the reward system based on goalkeeper and ball clearing actions
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage the goalkeeper to clear the ball effectively
            if o['active'] == 0 and o['ball_owned_player'] == 0:
                if o['game_mode'] in {2, 3, 4, 5, 6}:  # Consider ball clearance modes
                    reward[rew_index] += 0.1  # Small reward for holding the ball in pressure modes
                    components.setdefault("clearance_reward", []).append(0.1)
                else:
                    components.setdefault("clearance_reward", []).append(0.0)
            
            # Additional reward for passing the ball back to a safe player after recovery
            if o['game_mode'] == 0 and o['ball_owned_team'] == 0:  # Normal mode and own team controls the ball
                dist_to_goalie = np.linalg.norm(o['ball'][:2] - o['left_team'][0])  # Distance from ball to the goalie
                if dist_to_goalie < 0.1:  # Ball is very close to the goalkeeper
                    reward[rew_index] += 0.1
                    components.setdefault("goalie_proximity_reward", []).append(0.1)
                else:
                    components.setdefault("goalie_proximity_reward", []).append(0.0)
                
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
