import socket
import json


class Connection:
    def __init__(self, port):
        self.port = port

    def start(self, ip):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server.bind((ip, self.port))
        print(f"Waiting for client to connect on port {self.port}...")
        server.listen(1)

        self.client, _ = server.accept()
        print("Client connected.")

        init = json.loads(self.client.recv(2048).decode())
        self.action_count = init["actionCount"]
        self.observation_count = init["observationCount"]

        return self.action_count, self.observation_count

    def send(self, action):
        try:
            action_values = action.tolist()
        except AttributeError:
            action_values = action
        data = json.dumps({"action": action_values})
        self.client.send(data.encode())

    def recv(self):
        response = self.client.recv(4096)
        try:
            self.last_reaction = json.loads(response.decode())
        except json.decoder.JSONDecodeError as e:
            print(e, flush=True)
            self.last_reaction = {
                "observation": [0] * self.observation_count,
                "reward": 0,
                "terminated": False,
                "truncated": False,
            }
        return self.last_reaction
