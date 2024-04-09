import socket
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    # Get the host's IP address dynamically
    host_ip = socket.gethostbyname(socket.gethostname())
    app.run(host=host_ip, debug=True)
