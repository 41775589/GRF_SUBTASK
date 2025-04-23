import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward by emphasizing offensive skills such as passing, shooting,
    dribbling, and sprinting, which are essential in creating scoring opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the original reward to include additional rewards for 
        offensive actions that enable better scoring opportunities.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "sprinting_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, obs in enumerate(observation):
            if obs['game_mode'] == 0:  # Normal play mode
                # Reward short pass, long pass, and shot actions based on game context
                if obs['ball_owned_player'] == obs['active']:
                    if obs['sticky_actions'][1] or obs['sticky_actions'][2]:  # Short Pass or Long Pass
                        components['passing_reward'][i] = 0.05
                    if obs['sticky_actions'][6]:  # Shot
                        components['shooting_reward'][i] = 0.1

                    # Reward dribbling in control of the ball
                    if obs['sticky_actions'][9]:  # Dribble
                        components['dribbling_reward'][i] = 0.03

                # Reward sprinting when moving towards the opponent's goal
                if obs['sticky_actions'][8]:  # Sprint
                    target_goal_y = 0.42 if obs['ball'][1] > 0 else -0.42
                    if abs(obs['ball'][1] - target_goal_y) < 0.5:  # Close to the horizontal axis of the goal
                        components['sprinting_reward'][i] = 0.02

            # Combine the rewards
            total_reward = (reward[i] *
                            components['base_score_reward'][i] +
                            components['passing_reward'][i] + 
                            components['shooting_reward'][i] + 
                            components['dribbling_reward'][i] + 
                            components['sprinting_reward'][i])

            reward[i] = total_reward

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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
