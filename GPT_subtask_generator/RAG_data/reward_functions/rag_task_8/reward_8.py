import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for quick decision-making and efficient ball handling immediately after regaining possession."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.recent_possession = False
        self.ball_position_after_possession = None
        self.active_recovery_bonus = 0.5
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.recent_possession = False
        self.ball_position_after_possession = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['recent_possession'] = self.recent_possession
        to_pickle['ball_position_after_possession'] = self.ball_position_after_possession
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.recent_possession = from_pickle['recent_possession']
        self.ball_position_after_possession = from_pickle['ball_position_after_possession']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "recovery_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o['ball_owned_team']
            
            if ball_owned_team == 1:  # The right team (controlled by agents)
                if not self.recent_possession:  # Just regained possession
                    self.recent_possession = True
                    self.ball_position_after_possession = o['ball'].copy()
                else:
                    current_ball_position = o['ball']
                    original_position = self.ball_position_after_possession
                    if original_position is not None:
                        distance_moved = np.linalg.norm(current_ball_position[:2] - original_position[:2])
                        if distance_moved > 0.1:  # Encourage moving the ball forward quickly
                            components["recovery_bonus"][rew_index] = self.active_recovery_bonus
                            reward[rew_index] += components["recovery_bonus"][rew_index]
            else:  # Lost possession
                self.recent_possession = False
                self.ball_position_after_possession = None

        reward += self.env.reward(reward)
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
