from . import *
def build_scenario(builder):
    builder.config().game_duration = 3000
    builder.config().deterministic = False
    builder.config().offsides = True
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False

    # Set ball in a neutral position on the left team's side.
    builder.SetBallPosition(-0.5, 0.0)

    # Set up the left team with 1 agent in the Center Back role, focusing on defensive movement
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # Goalkeeper
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CB)  # Center Back, active agent training on positioning

    # Creating a simple scenario for the left team agent to practice intercepting passes
    # Set up right team passive players to recreate game-like scenarios for defensive positioning
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # Goalkeeper
    builder.AddPlayer(-0.2, 0.1, e_PlayerRole_CF, lazy=True)  # Center Forward
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_CF, lazy=True)  # Center Forward
    builder.AddPlayer(-0.3, 0.2, e_PlayerRole_CM, lazy=True)  # Center Midfielder
    builder.AddPlayer(-0.3, -0.2, e_PlayerRole_CM, lazy=True)  # Center Midfielder
