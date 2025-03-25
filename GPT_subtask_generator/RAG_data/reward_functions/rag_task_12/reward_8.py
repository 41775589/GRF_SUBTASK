import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that assigns rewards based on specific tasks related to 
    midfielder/advanced defender roles such as high pass, long pass, dribble, 
    sprint, and stop sprint abilities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'sticky_actions' in o:
                sprinting = o['sticky_actions'][8]  # Check if sprinting
                dribbling = o['sticky_actions'][9]  # Check if dribbling
                ball_owner = (o['ball_owned_team'] == 0 and 
                              o['ball_owned_player'] == o['active'])

                transition_bonus = 0
                if sprinting:
                    transition_bonus += 0.02  # Reward for effective sprinting
                if dribbling:
                    transition_bonus += 0.05  # Encourage dribbling under pressure

                if ball_owner:
                    if o['game_mode'] in {2, 5, 6}:  # Effective passes in different modes
                        transition_bonus += 0.1  # Reward for effective handling under game modes
                    
                components["transition_reward"][rew_index] = transition_bonus
                reward[rew_index] += transition_bonus

        return reward, components

    def step(self, action):
        observation, reward, done, info =  self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
                
        return observation, reward, done, info
