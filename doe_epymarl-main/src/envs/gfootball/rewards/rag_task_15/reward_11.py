import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that rewards technical skill in long passing. """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward_multiplier = 0.2  # Reward multiplier for successful passes
        self.distance_threshold = 0.3       # Minimum distance for a "long" pass
        self.precision_multiplier = 0.5     # Multiplier enhancing reward based on closeness to teammates

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
        # Initializing the components of reward for each agent
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}
        
        # Fetch the most recent environment observations
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Check if our team owns the ball
                last_ball_pos = o.get('last_ball_pos', o['ball'])  # Fetch last known ball position
                current_ball_pos = o['ball']
                
                # Compute the Euclidean distance the ball has traveled
                ball_travel_distance = np.linalg.norm(current_ball_pos[:2] - last_ball_pos[:2])
                
                if ball_travel_distance > self.distance_threshold:
                    components["long_pass_reward"][rew_index] = self.pass_reward_multiplier * ball_travel_distance
                                        
                    # Check how close the ball is to any teammate
                    for teammate_pos in o['left_team']:
                        distance_to_teammate = np.linalg.norm(teammate_pos - current_ball_pos[:2])
                        # Increase reward based on how close the ball is to teammates
                        if distance_to_teammate < 0.1:  # 0.1 threshold for being very close
                            components["long_pass_reward"][rew_index] += self.precision_multiplier / distance_to_teammate
                        
                    reward[rew_index] += components["long_pass_reward"][rew_index]
                
                o['last_ball_pos'] = current_ball_pos  # Update the last known position for the next step

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Track the sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_act
        return observation, reward, done, info
