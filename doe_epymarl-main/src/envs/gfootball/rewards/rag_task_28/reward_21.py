import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on dribbling skills against the goalkeeper."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_index = None  # index of the goalkeeper in the right_team
        self.dribbling_reward = 0.2
        self.close_control_bonus = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_index = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = None  # Update with necessary states if any
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            agent_position = o['left_team'][o['active']]
            goalkeeper_position = o['right_team'][o['goalkeeper_index'] if self.goalkeeper_index else -1]

            # Calculate distance to the goalkeeper
            if self.goalkeeper_index is None:
                # Identify the goalkeeper (usually the last player in the team list)
                self.goalkeeper_index = np.argmax(o['right_team_roles'] == 0)  # Role 0 is typically goalkeeper

            distance_to_goalkeeper = np.linalg.norm(agent_position - goalkeeper_position)
            
            # This part adjusts reward based on dribbling and close control when near the goalkeeper
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:  # Our agent has the ball
                if distance_to_goalkeeper < 0.1:  # Very close to the goalkeeper
                    components["dribbling_reward"][rew_index] += self.dribbling_reward

                    # Reward small movements to simulate dribbling
                    if np.any(o['sticky_actions'][6:10]):  # Check if slow movement actions are used
                        components["dribbling_reward"][rew_index] += self.close_control_bonus

            reward[rew_index] += components["dribbling_reward"][rew_index]

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
