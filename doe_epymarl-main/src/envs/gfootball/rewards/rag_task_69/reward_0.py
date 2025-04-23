import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defining the threshold on how close to the goal shots need to be for rewards
        self.shooting_threshold = 0.1
        # Defining rewards for successful dribbling, shooting or passing
        self.dribble_reward = 0.05
        self.shoot_reward = 0.1
        self.pass_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            base_reward = reward[rew_index]
            
            # Determine the rewards based on ball possession and actions performed
            if o['ball_owned_team'] == 1: # If the right team has the ball
                if o['sticky_actions'][9] == 1: # dribbling
                    components["dribble_reward"][rew_index] += self.dribble_reward
                    reward[rew_index] += self.dribble_reward

                if o['ball'][0] > (1 - self.shooting_threshold): # close enough to shoot
                    components["shoot_reward"][rew_index] += self.shoot_reward
                    reward[rew_index] += self.shoot_reward

                if o['sticky_actions'][8] == 1: # sprinting could indicate longer passes
                    components["pass_reward"][rew_index] += self.pass_reward
                    reward[rew_index] += self.pass_reward

            # Adjust the base reward
            reward[rew_index] += base_reward
            
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
