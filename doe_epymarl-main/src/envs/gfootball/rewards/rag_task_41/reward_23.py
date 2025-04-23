import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym wrapper for enhancing attacking skills in football by promoting finishing skills,
    creativity in offensive play, and adaptation to match-like pressures and defensive setups.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.attacking_skill_bonus = 0.2
        self.creative_play_bonus = 0.15
        self.pressure_handling_bonus = 0.1
        self.close_to_goal_threshold = 0.2  # Distance to goal threshold to consider 'close'
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions

    def reset(self):
        """
        Reset the environment state and sticky actions count.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the environment with rewards wrapper info appended.
        """
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment and extract rewards wrapper info from it.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modify obtained reward by adding bonuses for attacking skill factors.

        Adds bonuses for:
            - Engaging in successful attacking plays (close to opponent's goal).
            - Creative play demonstrated by using diverse actions.
            - Handling match pressure effectively.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "attacking_skill_bonus": 0.0,
                      "creative_play_bonus": 0.0,
                      "pressure_handling_bonus": 0.0}

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            obs = observation[i]
            ball_pos = obs['ball'][:2]
            ball_distance_to_goal = np.linalg.norm(ball_pos - [1, 0])  # Distance to the right goal
            
            # Check distance to goal to enhance attacking plays
            if ball_distance_to_goal < self.close_to_goal_threshold:
                components["attacking_skill_bonus"] += self.attacking_skill_bonus
                reward[i] += self.attacking_skill_bonus

            # Check use of creative plays
            if np.sum(obs['sticky_actions'][8:]) > 0:  # Checking if dribble or sprint are active
                components["creative_play_bonus"] += self.creative_play_bonus
                reward[i] += self.creative_play_bonus

            # Bonus for handling pressure: based on advancement in the field under defense
            if obs['ball_owned_team'] == 1 and ball_distance_to_goal < 0.5:
                components["pressure_handling_bonus"] += self.pressure_handling_bonus
                reward[i] += self.pressure_handling_bonus

        return reward, components

    def step(self, action):
        """
        Take a step using given 'action', enrich observation with detailed rewards breakdown.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = value

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for idx, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[idx] += action
                info[f"sticky_actions_{idx}"] = action

        return observation, reward, done, info
