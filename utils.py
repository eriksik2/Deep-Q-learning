import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=device).float()
    elif env in ['ALE/Pong-v5']:
        obs = torch.tensor(obs, device=device).float()

        # Rescale the observations from [0, 255] to [0, 1].
        obs /= 255.0
        return obs
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
