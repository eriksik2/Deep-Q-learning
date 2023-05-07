import os
import csv

class Recorder():
    def __init__(self, experiment_name, env_config):
        self.experiment_name = experiment_name
        self.env_config = env_config

        # List of (episode_num, episode_return) tuples.
        self.episode = []

    def record_episode(self, episode_num, episode_return):
        self.episode.append((episode_num, episode_return))

    def save(self, csv_path=None):
        if csv_path is None:
            csv_path = f"recorder/{self.experiment_name}.csv"

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "return"])

            for episode_num, episode_return in self.episode:
                writer.writerow([episode_num, episode_return])

        print(f"Saved results to {csv_path}.")