import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing offensive skills: passing, shooting, dribbling."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Define rewards for specific actions
        self.pass_reward = 0.03  # Reward for successful pass
        self.shot_reward = 0.5   # Reward for attempting a shot
        self.dribble_reward = 0.02  # Reward for dribbling
        self.sprint_reward = 0.01  # Reward for sprinting
        self.goal_score_reward = 1  # Reward for scoring a goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        pickle = self.env.get_state(to_pickle)
        pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Modify the rewards based on offensive gameplay elements."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "pass_reward": 0.0, "shot_reward": 0.0, 
                      "dribble_reward": 0.0, "sprint_reward": 0.0,
                      "goal_score_reward": 0.0}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Check for goal
            if o['score'][0] > self.env.unwrapped.previous_score[0]:  # Assuming index 0 is the scoring team
                reward[i] += self.goal_score_reward
                components["goal_score_reward"] += self.goal_score_reward

            # Rewards based on sticky actions and ball possession
            sticky_actions = o['sticky_actions']

            if o['ball_owned_team'] == 0:  # Assuming 0 is the team id of the agent
                if sticky_actions[7] or sticky_actions[8]:  # action_dribble indices
                    reward[i] += self.dribble_reward
                    components["dribble_reward"] += self.dribble_reward

                if sticky_actions[9]:  # action_sprint index
                    reward[i] += self.sprint_reward
                    components["sprint_reward"] += self.sprint_reward

                # Analyze attempts to pass or shoot
                game_modes = [3, 4]  # FreeKick and Corner which may involve passing or shooting
                if o['game_mode'] in game_modes:
                    if abs(o['ball_direction'][0]) > 0.5:  # assuming shot direction towards goal
                        reward[i] += self.shot_reward
                        components["shot_reward"] += self.shot_reward
                    else:
                        reward[i] += self.pass_reward
                        components["pass_reward"] += self.pass_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
