import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance crossing and sprinting abilities for wingers."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.crossing_reward = 1.0
        self.sprinting_reward = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward),
                      "sprinting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' not in o:
                continue
            
            active_player_role = o['right_team_roles'][o['active']] if o['ball_owned_team'] == 1 else o['left_team_roles'][o['active']]
            x_position = o['ball'][0]

            # Check if the wing player has the ball and is close to the crossing zones
            if active_player_role in [2, 6]:  # Check for left or right midfielders(wingers)
                # Wings defined by y positions near the edges
                if abs(o['ball'][1]) > 0.3:
                    if x_position > 0.7 or x_position < -0.7:  # Near the goal zones
                        components["crossing_reward"][rew_index] = self.crossing_reward
                        reward[rew_index] += components["crossing_reward"][rew_index]

            # Rewarding sprinting along the wings
            if o['sticky_actions'][8] == 1:  # Action_sprint is active
                if active_player_role in [2, 6] and abs(o['ball'][1]) > 0.2:
                    components["sprinting_reward"][rew_index] = self.sprinting_reward
                    reward[rew_index] += components["sprinting_reward"][rew_index]

            # Managing sticky actions counts
            for i, action in enumerate(o['sticky_actions']):
                self.sticky_actions_counter[i] += action

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
