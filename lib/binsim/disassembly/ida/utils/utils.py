import socket
import pickle
import struct
import select

START_MSG = b's'
PING_MSG = b'p'
PONG_MSG = b'o'
END_MSG = b'e'


def send_data(socket, data):
    data = pickle.dumps(data)
    socket.send(struct.pack('I', len(data)))
    socket.send(data)


def recv_data(socket):
    length = struct.unpack('I', socket.recv(4))[0]
    data = b''
    while len(data) < length:
        data += socket.recv(length - len(data))
    return pickle.loads(data)


class HeartbeatServer:
    def __init__(self, host='localhost', timeout=10, extra_data=None):
        self.host = host
        self.timeout = timeout
        self.server = None
        self.port = None
        self.extra_data = extra_data
        self._last_message = None

    def start(self, extra_data=None):
        assert self.server is None, "Server is already started!"
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, 0))
        self.port = self.server.getsockname()[1]
        self._last_message = None
        self.extra_data = extra_data

    @property
    def last_message(self):
        return self._last_message

    def monitor(self, proc):
        assert self.server is not None, "Server is not started!"
        self.server.listen(1)

        timeout = 5
        while True:
            ready_to_read, _, _ = select.select([self.server], [], [], timeout)
            if ready_to_read:
                client, addr = self.server.accept()
                break
            else:
                # check whether target process is still alive
                if proc.poll() is not None:
                    raise RuntimeError(f'The IDA process terminated abnormally.')

        msg = client.recv(1)
        if msg == START_MSG:
            client.send(START_MSG)
            send_data(client, self.extra_data)
        else:
            client.close()
            raise ValueError(f'Invalid message received: {msg}')
        client.settimeout(self.timeout)
        while True:
            try:
                msg = client.recv(1)
                if msg == PING_MSG:
                    self._last_message = recv_data(client)
                    client.send(PONG_MSG)
                    continue
                elif msg == END_MSG:
                    client.send(END_MSG)
                    break
                else:
                    client.close()
                    self.disconnect()
                    raise ValueError(f'Invalid message received:{msg}')
            except socket.timeout:
                client.close()
                self.disconnect()
                raise TimeoutError("The client seems to be stuck!")
        self.disconnect()

    def disconnect(self):
        self.server.close()
        self.server = None
        self.port = None


class HeartbeatClient:
    def __init__(self, host='localhost', port=0):
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.client.connect((self.host, self.port))
        self.client.send(START_MSG)
        msg = self.client.recv(1)
        if msg != START_MSG:
            self.client.close()
            raise ValueError(f'Invalid message received: {msg}')
        extra_data = recv_data(self.client)
        return extra_data

    def ping(self, msg):
        self.client.send(PING_MSG)
        send_data(self.client, msg)
        self.client.recv(1)

    def end(self):
        self.client.send(END_MSG)
        self.client.recv(1)
        self.client.close()
