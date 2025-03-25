import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that incentivizes quick decision-making and efficient ball handling
    to initiate counter-attacks after recovering possession. This reward encourages the
    player to move from their half to the opponent's half with possession quickly.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Counter for ball in player's own half
        self.player_half_steps = 0
        # Threshold to encourage fast play across the midfield line into the opponent's half
        self.fast_play_threshold = 10

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_half_steps = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['player_half_steps'] = self.player_half_steps
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickled = self.env.set_state(state)
        self.player_half_steps = from_pickled['player_half_steps']
        return from_pickled

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counter_attack_bonus": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            if o['ball_owned_team'] == 0:  # Assume 0 is the team of the agent
                ball_x = o['ball'][0]  # X-coordinates of the ball
                if ball_x <= 0:
                    self.player_half_steps += 1
                else:
                    self.player_half_steps = 0

                if self.player_half_steps > self.fast_play_threshold and ball_x > 0:
                    # Reward for moving to opponent's half quickly after regaining possession
                    components["counter_attack_bonus"][rew_index] = 0.5
                    reward[rew_index] += components["counter_attack_bonus"][rew_index]

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
            for i in range(len(agent_obs['sticky_actions'])):
                self.sticky_actions_counter[i] += agent_obs['sticky_actions'][i]
                info[f"sticky_actions_{i}"] = agent_obs['sticky_actions'][i]
        return observation, reward, done, info
