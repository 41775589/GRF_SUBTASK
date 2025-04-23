import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focusing on maintaining strategic positioning using all directional movements,
    ensuring effective pivoting between defensive stance and initiating attacks."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int) # count of each sticky action activated
        self.position_weight = 0.8  # Emphasis on maintaining positions
        self.attack_defense_switch = 0.2  # Weight for changing strategies from attack to defense and vice versa

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['StickyActionsCounter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['StickyActionsCounter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "strategy_switch_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["positioning_reward"][rew_index] = 0
            components["strategy_switch_reward"][rew_index] = 0

            # Encourage maintaining effective positioning
            active_position = np.array(o['left_team' if o['active'] in o['left_team_roles'] else 'right_team'][o['active']])
            goal_position = np.array([-1, 0]) if o['active'] in o['left_team_roles'] else np.array([1, 0])
            distance_to_goal = np.linalg.norm(active_position - goal_position)
            components["positioning_reward"][rew_index] = distance_to_goal * self.position_weight

            # Reward for switching between defending and attacking based on the game state
            current_mode = o['game_mode']
            if ((current_mode == 2 or current_mode == 3) and o['ball_owned_team'] == 0) or (
                current_mode == 4 and o['ball_owned_team'] == 1):
                # From defending to attacking or attacking to defending
                components["strategy_switch_reward"][rew_index] = self.attack_defense_switch

            # Update overall reward
            reward[rew_index] += components["positioning_reward"][rew_index] + components["strategy_switch_reward"][rew_index]

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
        return observation, reward, done, info
