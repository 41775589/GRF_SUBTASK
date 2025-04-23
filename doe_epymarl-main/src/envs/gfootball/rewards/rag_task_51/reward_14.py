import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A Gym wrapper that focuses on rewards specific for training a goalkeeper in football.
    It adds rewards for shot stopping, reflex actions, and initiating counter-attacks."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_stop_reward = 0.5
        self.reflex_bonus = 0.3
        self.passing_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Get the current observation to determine the state of the game and the goalkeeper
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_stop_reward": [0.0] * len(reward),
                      "reflex_bonus": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            # Assumption: the goalkeeper is at index 0 in left team for simplicity
            goalkeeper_pos = o['left_team'][0]
            ball_pos = o['ball'][:2]  # Ignore ball's z position
            
            # Shot-stopping reward: if the goalkeeper is close to the ball when the opponent shoots
            if (o['ball_owned_team'] == 1 and np.linalg.norm(ball_pos - goalkeeper_pos) < 0.1):
                components["shot_stop_reward"][rew_index] = self.shot_stop_reward

            # Reflex bonus: if goalkeeper's movement is quick, inferred by high change in their direction
            goal_keeper_dir = np.linalg.norm(o['left_team_direction'][0])
            if goal_keeper_dir > 0.01:
                components["reflex_bonus"][rew_index] = self.reflex_bonus
            
            # Passing reward: if the ball is passed by goalkeeper under pressure
            if (o['ball_owned_team'] == 0 and o['ball_owned_player'] == 0 and
                np.any(o['sticky_actions'][0:2])):  # Assuming actions 0,1 can be related to passing
                components["passing_reward"][rew_index] = self.passing_reward

            reward[rew_index] += (components["shot_stop_reward"][rew_index] +
                                  components["reflex_bonus"][rew_index] +
                                  components["passing_reward"][rew_index])

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
                self.sticky_actions_counter[i] += int(action)
        return observation, reward, done, info
