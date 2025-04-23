import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for successful counterattacks which 
    involves making accurate long passes from defense to attack.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Thresholds and weights could be adjusted as necessary
        self.long_pass_distance = 0.5  # This means that a long pass must cross at least half the field width
        self.transition_reward = 1.25  # A successful transition yields this reward

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": reward.copy(),
            "transition_reward": [0.0, 0.0]  # Assuming two players/agents
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation), "Mismatch in the number of agents and rewards received"

        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Check if the left team (defending team) had the ball and successfully made a long pass transitioning to an attack
            if o['ball_owned_team'] == 0:
                initial_ball_position = np.array(o['ball'])
                post_pass_ball_position = initial_ball_position + np.array(o['ball_direction'])

                # Check if the ball has crossed the long pass distance threshold and moved towards the attacking side
                if post_pass_ball_position[0] - initial_ball_position[0] > self.long_pass_distance:
                    components["transition_reward"][rew_index] = self.transition_reward
                    reward[rew_index] += components["transition_reward"][rew_index]

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
