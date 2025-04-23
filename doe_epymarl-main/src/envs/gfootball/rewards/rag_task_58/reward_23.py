import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for mastering defensive strategies and efficient 
    transitions from defense to attack.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.past_ball_position = None
        self.defensive_efficiency_bonus = 0.1
        self.transition_efficiency_bonus = 0.2
    
    def reset(self):
        """
        Reset the environment and variables.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.past_ball_position = None
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """
        Get the state of the environment with additional data.
        """
        to_pickle['past_ball_position'] = self.past_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment with additional data.
        """
        from_picle = self.env.set_state(state)
        self.past_ball_position = from_picle['past_ball_position']
        return from_picle

    def reward(self, reward):
        """
        Calculate additional rewards based on defensive plays and transitions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_play_bonus": [0.0] * len(reward),
                      "transition_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        current_ball_position = observation[0]['ball']

        # Reward for intercepting the ball or blocking a shot
        for i, o in enumerate(observation):
            if self.past_ball_position is not None:
                ball_moved_toward_own_goal = (
                    self.past_ball_position[0] < current_ball_position[0]
                    if o['left_team_tired_factor'][0] < 0.5 else
                    self.past_ball_position[0] > current_ball_position[0]
                )

                if ball_moved_toward_own_goal and o['ball_owned_team'] == 0:
                    components["defensive_play_bonus"][i] += self.defensive_efficiency_bonus
                    reward[i] += 1.1 * components["defensive_play_bonus"][i]

        # Reward for successful transitions:
        if self.past_ball_position is not None:
            distance_covered = np.linalg.norm(self.past_ball_position[:2] - current_ball_position[:2])
            if distance_covered > 0.5:
                components["transition_bonus"] += [self.transition_efficiency_bonus] * len(reward)
                reward[i] += 1.2 * components["transition_bonus"][i]

        self.past_ball_position = current_ball_position
        
        return reward, components

    def step(self, action):
        """
        Execute a step using the action provided, and return adjusted observations,
        rewards, component details, and overall info.
        """
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
