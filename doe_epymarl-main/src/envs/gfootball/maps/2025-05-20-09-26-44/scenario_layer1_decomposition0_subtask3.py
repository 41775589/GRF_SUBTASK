from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set the initial ball position to ensure it starts near the player.
    builder.SetBallPosition(0.5, 0.0)

    # Setting up the left team (player-controlled agents)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB, controllable=True)  # Centre Back who can pass high

    # Setting up the right team (AI-controlled opponents)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper    
    builder.AddPlayer(0.5, 0.25, e_PlayerRole_CF)  # Opponent Centre Forward
    builder.AddPlayer(0.5, -0.25, e_PlayerRole_CF)  # Opponent Centre Forward
    builder.AddPlayer(0.5, 0, e_PlayerRole_CM)  # Opponent Centre Midfielder

    # This setting creates a scenario where the player's agent can practice high passes
    # towards forward positions effectively, and also focus on good positioning to intercept direct attacks.
