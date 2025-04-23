import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for executing successful high passes from the midfield which create 
    direct scoring opportunities, emphasizing both precision and participation in creating chances.
    """
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pass_reward = 0.5  # Reward for successful high passes from the midfield
        self._scoring_chance_reward = 1.0  # Additional reward if the high pass leads to a direct scoring opportunity

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Store additional states if necessary
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Retrieve additional states if necessary
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward),
            "scoring_chance_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for index in range(len(reward)):
            o = observation[index]
            
            if ('ball_owned_team' not in o or o['ball_owned_team'] != 0):
                continue  # Ensuring possession by the left team (assuming the agents' team)

            ball_pos = o['ball']
            ball_dir = o['ball_direction']
            
            # Check if the ball is in midfield and moving upward towards the opponent's goal area
            if -0.2 <= ball_pos[0] <= 0.2 and ball_dir[1] > 0 and o['game_mode'] == 0:  # Game mode 0 is Normal play
                # Assuming a high ball is defined by a larger z component in direction or a specific action
                if ball_dir[2] > 0.1:  # High pass threshold
                    components["high_pass_reward"][index] = self._high_pass_reward
                    # Reward for simply executing the high pass in midfield
            
            # Check for direct scoring opportunity creation
            if ball_dir[0] > 0.3 and abs(ball_pos[1]) < 0.04 and ball_dir[2] > 0.1:  # Direct alignment towards the goal
                components["scoring_chance_reward"][index] = self._scoring_chance_reward
            
            # Calculate the total reward for this agent
            reward[index] += components["high_pass_reward"][index]
            reward[index] += components["scoring_chance_reward"][index]

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
