from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    # Ball starts near the defensive line, simulating a scenario of possession transfer from defense to midfield
    builder.SetBallPosition(-0.8, 0.0)

    builder.SetTeam(Team.e_Left)
    # Add goalkeeper
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Add the single agent training in this subtask - Positioned as a Defensive Midfielder to master passing linkages
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_DM)

    # Set up a simple opposition to contest ball possession and create realistic game pressure
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Multiple opposing players to simulate defensive pressure and passing interception attempts
    builder.AddPlayer(-0.6, 0.1, e_PlayerRole_CM)
    builder.AddPlayer(-0.6, -0.1, e_PlayerRole_CM)
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CM)
