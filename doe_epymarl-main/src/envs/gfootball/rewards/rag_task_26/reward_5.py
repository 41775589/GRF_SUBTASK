import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward system focusing on midfield dynamics 
    pertaining to both defense and offensive transitions through midfield roles."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize metrics for midfield control
        self.midfield_control_count = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_count = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control": [0.0] * len(reward)}
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Example: Implementing midfield dynamics recognition based on ball position
            # and midfield players' active engagements. Tailor with specific midfield mechanics as needed.
            midfield_zone = (-0.3, 0.3)  # Define central midfield zone as example
            if o['ball'][0] > midfield_zone[0] and o['ball'][0] < midfield_zone[1]:
                if o['ball_owned_team'] == 0 and o['left_team_roles'][o['active']] in [3, 4, 5, 6]:
                    # Assuming roles 3,4,5,6 correspond to central and wide midfielders
                    components["midfield_control"][rew_index] = 0.5
                elif o['ball_owned_team'] == 1 and o['right_team_roles'][o['active']] in [3, 4, 5, 6]:
                    components["midfield_control"][rew_index] = 0.5
            
            reward[rew_index] += components["midfield_control"][rew_index]

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'midfield_control_count': self.midfield_control_count,
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        loaded = from_pickle['CheckpointRewardWrapper']
        self.midfield_control_count = loaded['midfield_control_count']
        self.sticky_actions_counter = np.array(loaded['sticky_actions_counter'], dtype=int)
        return from_pickle

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
