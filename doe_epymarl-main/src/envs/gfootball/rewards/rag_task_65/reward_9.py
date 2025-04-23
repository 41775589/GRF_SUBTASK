import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic shooting and passing reward based on player positions and ball control."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward_value = 0.3
        self.shooting_reward_value = 0.5
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Encourage passing by checking for changes in ball ownership among team members
            if o['ball_owned_team'] == 0:  # Check if ball is owned by agent's team
                if 'previous_ball_owner' in o and o['ball_owned_player'] != o['previous_ball_owner']:
                    components["passing_reward"][rew_index] = self.passing_reward_value
                    reward[rew_index] += components["passing_reward"][rew_index]

            # Encourage shooting by rewarding attempts near the goal
            if o['ball_owned_team'] == 0:
                # Dist from opponent's goal (x=1 is goal location for opponent)
                dist_to_goal = 1 - o['ball'][0] 
                if dist_to_goal < 0.2:
                    components["shooting_reward"][rew_index] = self.shooting_reward_value
                    reward[rew_index] += components["shooting_reward"][rew_index]

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
