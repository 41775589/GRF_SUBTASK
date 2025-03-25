import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward focused on offensive strategies."""

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        maintain_ball_possession_bonus = 0.01
        effective_pass_bonus = 0.02
        accurate_shoot_bonus = 0.05
        dribble_bonus = 0.02

        for rew_index, o in enumerate(observation):
            # Encourage keeping possession of the ball
            if o["ball_owned_team"] == 0:  # Assuming the agent's team is 0
                components.setdefault("possession_bonus", [0] * len(reward))
                components["possession_bonus"][rew_index] += maintain_ball_possession_bonus

            # Encourage shooting at the goal effectively
            if o["game_mode"] == 6 and o["ball_owned_team"] == 0:  # Penalty mode as a proxy for shooting attempts
                components.setdefault("shoot_bonus", [0] * len(reward))
                components["shoot_bonus"][rew_index] += accurate_shoot_bonus
            
            # Encourage effective dribbling
            if o["sticky_actions"][9] == 1:  # dribble is active
                components.setdefault("dribble_bonus", [0] * len(reward))
                components["dribble_bonus"][rew_index] += dribble_bonus
            
            # Reward successful and strategically important passes
            if o["game_mode"] in [3, 4]:  # Freekick or Corner as a proxy for successful passes
                components.setdefault("pass_bonus", [0] * len(reward))
                components["pass_bonus"][rew_index] += effective_pass_bonus

        # Calculate final reward
        for idx in range(len(reward)):
            reward[idx] += sum([components[key][idx] for key in components.keys() if key != "base_score_reward"])

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
