import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to augment defensive play and goalkeeping in football agents."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defense_rewards = {}
        self.goalkeeping_rewards = {}
        self.defensive_zones = 3  # Divides the field into 3 defense-oriented zones
        self.goalkeeper_bonus = 0.3  # Bonus for goalkeeping actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defense_rewards = {}
        self.goalkeeping_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defense_rewards'] = self.defense_rewards
        to_pickle['goalkeeping_rewards'] = self.goalkeeping_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defense_rewards = from_pickle['defense_rewards']
        self.goalkeeping_rewards = from_pickle['goalkeeping_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "defensive_play_reward": [0.0] * len(reward),
                      "goalkeeping_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, agent_obs in enumerate(observation):
            current_position = agent_obs['left_team'][agent_obs['active']]

            # Assign rewards based on proximity to own goal
            distance_to_goal = np.linalg.norm(current_position - [-1, 0])
            zone_thresholds = np.linspace(0, 1, self.defensive_zones + 1)
            zone_reached = np.digitize([distance_to_goal], zone_thresholds)[-1]

            if agent_obs['left_team_roles'][agent_obs['active']] == 0:
                # Goalkeeper specific actions
                if agent_obs['ball_owned_player'] == agent_obs['active']:
                    if zone_reached in self.goalkeeping_rewards and self.goalkeeping_rewards[zone_reached] == 0:
                        components["goalkeeping_reward"][rew_index] = self.goalkeeper_bonus
                        self.goalkeeping_rewards[zone_reached] = 1

            else:
                # Defense rewards based on how often they recover the ball in their zones
                if agent_obs['ball_owned_player'] == agent_obs['active']:
                    if zone_reached not in self.defense_rewards:
                        self.defense_rewards[zone_reached] = 0
                    if self.defense_rewards[zone_reached] < zone_reached:
                        components["defensive_play_reward"][rew_index] = 0.1 * zone_reached
                        self.defense_rewards[zone_reached] += 1

            reward[rew_index] += (
                components["defensive_play_reward"][rew_index] +
                components["goalkeeping_reward"][rew_index]
            )

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
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
