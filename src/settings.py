
import json

class Settings:
    def __init__(self, path='settings.json'):
        self.path = path
        self.config = {}

    def load(self):
        try:
            with open(self.path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")
