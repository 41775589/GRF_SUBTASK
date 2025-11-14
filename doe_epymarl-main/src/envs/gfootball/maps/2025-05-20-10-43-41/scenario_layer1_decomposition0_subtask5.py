from . import *
def build_scenario(builder):
    # Set the overall game configurations
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set the initial position of the ball, somewhat closer to the center, to simulate incoming ground passes
    builder.SetBallPosition(0.2, 0.0)

    # Define the team on the left (team of the trained agent)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB, controllable=True)  # Center Back (trained agent)

    # Define the team on the right (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.3, -0.1, e_PlayerRole_CF)  # Opposing team's center forward, positioned to attempt passes
    builder.AddPlayer(0.3, 0.1, e_PlayerRole_CF)  # Another forward positioned symmetrically to simulate central attacks
