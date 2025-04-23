import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive-oriented reward based on preventing scoring and strategic counterattacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_distance = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_distance = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['previous_ball_distance'] = self.previous_ball_distance
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_distance = from_pickle['previous_ball_distance']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        # Defining defensive reward component based on the opposition's progress towards the goal
        defensive_rewards = np.zeros(len(reward))

        for rew_index, o in enumerate(observation):
            if o['game_mode'] in [0]:  # Play is normal
                ball_position = o['ball'][:2]
                goal_position = [1, 0] if o['ball_owned_team'] == 1 else [-1, 0]
                current_ball_distance = np.linalg.norm(np.array(ball_position) - np.array(goal_position))
                
                if self.previous_ball_distance is None:
                    self.previous_ball_distance = current_ball_distance
                
                if goal_position[0] > 0:
                    # Right side is opponent's goal
                    if ball_position[0] > 0.5:  # Ball is on the opponent's side
                        defensive_rewards[rew_index] += 0.05
                else:
                    # Left side is opponent's goal
                    if ball_position[0] < -0.5:  # Ball is on the opponent's side
                        defensive_rewards[rew_index] += 0.05
                
                # Reward for reducing the proximity to own goal from previous
                if current_ball_distance > self.previous_ball_distance:
                    defensive_rewards[rew_index] += 0.1
            
            # Update previous distance after computing rewards
            self.previous_ball_distance = current_ball_distance

        for idx in range(len(reward)):
            reward[idx] += defensive_rewards[idx]
        
        components['defensive_rewards'] = defensive_rewards
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
