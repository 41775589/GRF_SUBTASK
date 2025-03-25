import gym
import numpy as np
class OffensiveStrategyRewardWrapper(gym.RewardWrapper):
    """
    This reward wrapper focuses on improving offensive strategies,
    including accurate shooting, effective dribbling, and mastering different pass types.
    """
    
    def __init__(self, env):
        super(OffensiveStrategyRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_reward = 0.1
        self.shot_accuracy_reward = 0.2
        self.dribbling_skill_reward = 0.15

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
        """
        Enhance the reward function with rewards focused on offensive gameplay:
        - Increase reward for successful passes.
        - Further reward for goal attempts/shots, more for goals.
        - Reward maintaining possession and successful dribbles.
        """
        observation = self.env.unwrapped.observation()  # accessing the raw observations of the environment
        components = {"base_score_reward": np.array(reward, copy=True),
                      "pass_accuracy_reward": [0.0] * len(reward),
                      "shot_accuracy_reward": [0.0] * len(reward), 
                      "dribbling_skill_reward": [0.0] * len(reward)}

        for idx, o in enumerate(observation):
            player_has_ball = (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'])

            if player_has_ball:
                # Reward based on sticky actions for dribbling and sprinting
                if o['sticky_actions'][8] or o['sticky_actions'][9]:  # action_dribble or action_sprint
                    components['dribbling_skill_reward'][idx] = self.dribbling_skill_reward
                    reward[idx] += components['dribbling_skill_reward'][idx]

            score_goal = (o['score'][0] > 0)  # Assuming the agent's team is 'left_team'
            if score_goal:
                components['shot_accuracy_reward'][idx] = self.shot_accuracy_reward
                reward[idx] += components['shot_accuracy_reward'][idx]

            # Simulating a successful pass by checking ball movement to another player
            if 'successful_pass' in o and o['successful_pass']:
                components['pass_accuracy_reward'][idx] = self.pass_accuracy_reward
                reward[idx] += components['pass_accuracy_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        return observation, reward, done, info
