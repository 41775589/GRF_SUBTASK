from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.0, 0.0)  # Start the ball in the center of the field

    # Setting up the left team (our team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Add the goalkeeper
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB)  # Add our agent focused on Sliding tackles

    # Setting up the right team (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Add the opponent's goalkeeper
    # Adding multiple opponents to practice tackling against
    builder.AddPlayer(-0.4, 0.1, e_PlayerRole_CF, lazy=True)
    builder.AddPlayer(-0.4, -0.1, e_PlayerRole_CF, lazy=True)
    builder.AddPlayer(-0.3, 0.2, e_PlayerRole_CM, lazy=True)
    builder.AddPlayer(-0.3, -0.2, e_PlayerRole_CM, lazy=True)
