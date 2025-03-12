import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward function focused on offensive strategies including accurate shooting, 
    effective dribbling, and practicing different pass types."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.passing_skill_multiplier = 0.5
        self.shooting_skill_multiplier = 1.0
        self.dribbling_skill_multiplier = 0.3

    def reset(self):
        self._collected_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['collected_rewards'] = self._collected_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_rewards = from_pickle.get('collected_rewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:
                # Calculate additional rewards for game skills
                if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                    own_player = o['right_team'][o['active']]
                    opponent_goal = [1, 0]  # Assuming the goal is at X=1, Y=0
                    dist_to_goal = np.linalg.norm(own_player - opponent_goal)
                    
                    # Reward for getting closer to the opponent's goal with the ball
                    components["shooting_reward"][rew_index] = self.shooting_skill_multiplier * (1 - dist_to_goal)
                    
                    # Reward for dribbling (encourage sticky actions like dribble)
                    components["dribbling_reward"][rew_index] = self.dribbling_skill_multiplier * int(o['sticky_actions'][9])

                    # Reward for successful passes (simple model based on passing actions, could be more complex)
                    components["passing_reward"][rew_index] = self.passing_skill_multiplier * (int(o['sticky_actions'][4]) + int(o['sticky_actions'][5]))

            # Update the actual reward array for the agent
            reward[rew_index] += components["shooting_reward"][rew_index]
            reward[rew_index] += components["dribbling_reward"][rew_index]
            reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
