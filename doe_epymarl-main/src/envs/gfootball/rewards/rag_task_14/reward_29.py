import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on rewarding the actions of a 'sweeper' in a football game."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_last_ball_position'] = self._last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_ball_position = from_pickle.get('CheckpointRewardWrapper_last_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        components["defense_bonus"] = 0.0
        components["ball_clearance_bonus"] = 0.0

        for o in observation:
            if o['score'][0] != o['score'][1]:  # Reward only when the score is tied.
                continue
            ball_pos = o['ball'][:2]
            if self._last_ball_position is not None:
                ball_movement = np.linalg.norm(self._last_ball_position - ball_pos)
            else:
                ball_movement = 0

            # Checks if the ball is in the defensive half and controlled by the team.
            if ball_pos[0] <= 0 and o['ball_owned_team'] == 0:
                components["defense_bonus"] += 0.1  # encourage being ready to defend

            # Checks if the ball is cleared to the opponent's half while previously near the goal
            if ball_pos[0] > 0 and self._last_ball_position is not None and self._last_ball_position[0] <= -0.4:
                components["ball_clearance_bonus"] += 1.0  # reward clearing the ball

            self._last_ball_position = ball_pos  # update the last known position of the ball

        for component in components:
            reward += components[component]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = value

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
