import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering defensive strategies and efficient transition play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Default components of the reward with the initial scores
        components = {
            "base_score_reward": np.array(reward),
            "defensive_play_reward": np.zeros(len(reward)),
            "transition_efficiency_reward": np.zeros(len(reward))
        }

        observation = self.env.unwrapped.observation()
        if not observation:
            return reward, components

        for idx, obs in enumerate(observation):
            # Defensive reward: Encourage the player not to lose the ball close to own goal.
            if obs['ball_owned_team'] == 0:  # ball owned by our team
                ball_pos_x = obs['ball'][0]
                if ball_pos_x < -0.5:  # closer to own goal
                    components["defensive_play_reward"][idx] += 0.1

            # Transition efficiency reward: Promote fast switching from defense to attack
            if obs['game_mode'] in [2, 3]:  # from defense mechanisms like GoalKick or FreeKick
                ball_direction = np.linalg.norm(obs['ball_direction'][:2])
                if ball_direction > 1.0:  # if the ball is moving fast towards the opponent's half
                    components["transition_efficiency_reward"][idx] += 0.1

        # Combine the rewards with the components
        total_reward = components["base_score_reward"] + \
                       components["defensive_play_reward"] + \
                       components["transition_efficiency_reward"]
                       
        return total_reward.tolist(), components

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
