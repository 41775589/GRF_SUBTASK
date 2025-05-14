from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False

    # Position the ball for different shots
    if builder.EpisodeNumber() % 3 == 0:
        builder.SetBallPosition(0.5, 0.0)  # Center, closer to the goal
    elif builder.EpisodeNumber() % 3 == 1:
        builder.SetBallPosition(0.5, 0.3)  # Left side
    else:
        builder.SetBallPosition(0.5, -0.3)  # Right side

    # Set up the teams
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goal keeper for the left team
    builder.AddPlayer(0.5, 0.0, e_PlayerRole_CF)  # A center forward for shooting practice

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goal keeper for the right team
    # Add a defensive player in front of the shooter to simulate defense
    if builder.EpisodeNumber() % 2 == 0:
        builder.AddPlayer(0.4, 0.05, e_PlayerRole_CB)
    else:
        builder.AddPlayer(0.4, -0.05, e_PlayerRole_CB)
