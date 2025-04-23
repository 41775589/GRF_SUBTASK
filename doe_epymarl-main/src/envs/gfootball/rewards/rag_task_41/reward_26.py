import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward function focusing on attacking play and creating offensive opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize parameters to encourage creative offensive play
        self.goal_approach_reward = 0.1  # Reward for moving towards the opponent's goal
        self.shot_on_goal_reward = 1.0   # Large reward for shots towards goal
        self.pass_towards_goal_reward = 0.5  # Reward for passes that progress towards the opponent's goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_stick_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_stick_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_approach_reward": [0.0] * len(reward),
                      "shot_on_goal_reward": [0.0] * len(reward),
                      "pass_towards_goal_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
            
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            goal_pos = [1, 0] if o['ball_owned_team'] == 0 else [-1, 0]
            goal_vector = np.subtract(goal_pos, player_pos)
            ball_vector = np.subtract(o['ball'][:2], player_pos)

            # Reward players moving towards the goal with the ball
            if np.dot(goal_vector, ball_vector) > 0:  # Positive if ball is moving towards opponent's goal
                components["goal_approach_reward"][rew_index] = self.goal_approach_reward
                reward[rew_index] += components["goal_approach_reward"][rew_index]

            # Check and reward shots on goal
            if o['ball_owned_team'] == o['ball_owned_player'] and o['game_mode'] in [2, 4]:  # Shot attempt
                components["shot_on_goal_reward"][rew_index] = self.shot_on_goal_reward
                reward[rew_index] += components["shot_on_goal_reward"][rew_index]

            # Reward successful passes towards goal direction
            if o['sticky_actions'][6] == 1:  # Pass action index 6
                if np.dot(goal_vector, ball_vector) > 0.1:  # Pass is directed towards goal
                    components["pass_towards_goal_reward"][rew_index] = self.pass_towards_goal_reward
                    reward[rew_index] += components["pass_towards_goal_reward"][rew_index]
        
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
