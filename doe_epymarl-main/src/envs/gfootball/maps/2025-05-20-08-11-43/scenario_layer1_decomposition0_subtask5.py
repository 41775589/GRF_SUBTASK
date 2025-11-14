from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.5, 0.0)  # Setting the ball in a central position to facilitate passing

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CM)  # Trained player placed in a central midfielder role to practice passes

    # Setting up a simple opposition to challenge the player
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.5, -0.2, e_PlayerRole_CB)  # A defender on the right side
    builder.AddPlayer(-0.5, 0.2, e_PlayerRole_CB)  # A defender on the left side
    builder.AddPlayer(-0.2, 0.0, e_PlayerRole_DM)  # A defensive midfielder to press the ball
