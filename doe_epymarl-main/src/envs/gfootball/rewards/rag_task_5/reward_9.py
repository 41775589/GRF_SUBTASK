import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive training and quick counter-attacks in agents."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions": self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle.get('CheckpointRewardWrapper', {}).get("sticky_actions", []), dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for i, single_reward in enumerate(reward):
            # Extract observations for both teams
            obs = observation[i]
            components.setdefault("defensive_positioning_reward", [])
            components.setdefault("quick_transition_reward", [])

            if obs['ball_owned_team'] == 0:  # if ball is possessed by our team
                own_player_pos = obs['left_team'][obs['active']]
                opponent_goal = np.array([1, 0])  # position of opponent goal

                if 'right_team' in obs:
                    for opponent in obs['right_team']:
                        # Reward based on distance between our player with the ball and nearest opponent
                        dist_to_opponent = np.linalg.norm(own_player_pos - opponent)
                        if dist_to_opponent < 0.1:
                            components["defensive_positioning_reward"].append(0.1)
                            single_reward += 0.1  # Better reward if maintaining closer marking

            if obs['ball_owned_team'] == 1:  # ball possession is with the opponent
                own_goal = np.array([-1, 0])  # own goal position
                ball_position = obs['ball'][:2]

                # Reward for quick transition to defense on losing ball possession
                dist_to_ball = np.linalg.norm(own_goal - ball_position)
                if dist_to_ball >= 0.5:
                    components["quick_transition_reward"].append(0.3)
                    single_reward += 0.3

            reward[i] = single_reward

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
