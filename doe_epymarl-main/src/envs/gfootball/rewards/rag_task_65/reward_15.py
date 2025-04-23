import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for precision shooting and passing in scenario-based training."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for precision and strategic targeting
        self.shooting_threshold = 0.9  # Close to goal increases precision bonus
        self.passing_bonus = 0.1  # Bonus for successful passes in the opponent's half
        self.shooting_bonus = 0.5  # Bonus for shots taken close to the goal within the threshold

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.copy()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "precision_shooting_bonus": [0.0] * len(reward), 
                      "strategic_passing_bonus": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Precision based on goal proximity
            if o['game_mode'] == 0:  # only consider normal gameplay mode
                ball_pos = o['ball'][:2]  # get x, y coordinates
                if abs(ball_pos[0]) > self.shooting_threshold:  # ball is close to a goal
                    components["precision_shooting_bonus"][rew_index] = self.shooting_bonus
                    reward[rew_index] += components["precision_shooting_bonus"][rew_index]
                
                # Bonus for successful passes in the opponent's half
                if ('ball_owned_team' in o and o['ball_owned_team'] == 1 and ball_pos[0] > 0) or \
                   ('ball_owned_team' in o and o['ball_owned_team'] == 0 and ball_pos[0] < 0):
                    components["strategic_passing_bonus"][rew_index] = self.passing_bonus
                    reward[rew_index] += components["strategic_passing_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
