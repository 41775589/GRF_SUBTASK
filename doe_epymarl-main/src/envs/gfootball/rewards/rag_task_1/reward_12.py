import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for mastering offensive maneuvers and adapting to game phases."""

    def __init__(self, env):
        super().__init__(env)
        self.offense_reward = 0.2
        self.game_phase_adaptation_reward = 0.1
        self.previous_ball_position = np.array([0, 0, 0])  # Initial assumed ball position
        self.quick_attack_threshold = 20  # Number of steps in which quick attack transition is expected
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.previous_ball_position = np.array([0, 0, 0])  # Reset ball position on env reset
        self.sticky_actions_counter.fill(0)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offense_reward": [0.0] * len(reward),
                      "adaptation_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            ball_position = o['ball']

            # Incentivize moving the ball towards the opponent's goal
            if o['ball_owned_team'] == 1:  # If right team owns the ball
                ball_progress = self.previous_ball_position[0] - ball_position[0]
                if ball_progress > 0:  # Positive progress towards left side goal
                    components["offense_reward"][rew_index] = self.offense_reward * ball_progress

            self.previous_ball_position = ball_position
            current_game_mode = o['game_mode']
            
            # Reward adapting quickly to game mode changes, particularly in offensive situations
            if current_game_mode in [2, 3, 4]:  # Handling special game phases such as FreeKick, GoalKick, or Corner
                components["adaptation_reward"][rew_index] = self.game_phase_adaptation_reward

            reward[rew_index] = reward[rew_index] + components["offense_reward"][rew_index] + components["adaptation_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Compute adjusted reward and components
        reward, components = self.reward(reward)
        # Track the metrics in info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counter for educational purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        # Return potentially augmented information
        return observation, reward, done, info
