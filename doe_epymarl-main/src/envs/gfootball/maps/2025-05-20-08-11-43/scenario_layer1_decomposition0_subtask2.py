from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
  
    # Set the ball in a typical midfield position
    builder.SetBallPosition(0.0, 0.0)

    # Setting team to be defensive and intercept passes
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper stays at the goal
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB)  # A single center back trained to intercept

    # Right team attackers trying to challenge interception
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent's goalkeeper
    builder.AddPlayer(-0.2, 0.15, e_PlayerRole_CF)  # Forward players positioned to attempt passes around the CB
    builder.AddPlayer(-0.2, -0.15, e_PlayerRole_CF)
