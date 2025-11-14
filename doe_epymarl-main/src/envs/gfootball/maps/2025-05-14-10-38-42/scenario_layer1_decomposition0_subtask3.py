from . import *
def build_scenario(builder):
    builder.config().game_duration = 300
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False

    # Set the initial ball position for the shooting scenario
    builder.SetBallPosition(0.5, 0.0)

    # Setting up the Left team (our team with trainee player)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Add a goalkeeper
    builder.AddPlayer(0.5, 0.0, e_PlayerRole_CF)   # Add our solo agent playing center forward, close to the ball

    # Setting up the Right team (opponent team with a goalkeeper)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Add goalkeeper for the right team
