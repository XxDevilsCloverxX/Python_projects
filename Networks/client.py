"""
    @author: Silas Rodriguez
    @program: client.py
    @brief: TCP Client for receiving data from a listening port on the server
    @date: 11/20/2023
"""

import socket   # Used for creating and managing sockets
import argparse # Used for parsing command line arguments
import json     # Used for JSON encoding/decoding

"""
    @brief: Receives and processes data from the connected server socket.
    @param: server_socket - Socket object for the connected server
    @return: None
    @pre: The server socket is connected and actively sending data
    @post: Receives and processes data from the server socket
"""
def receive_data(server_socket):
    buffer = ''

    while True:
        chunk = server_socket.recv(4096)
        if not chunk:
            break

        buffer += chunk.decode()

        try:
            # Try to decode JSON objects from the buffer
            while '{' in buffer and '}' in buffer:
                start_idx = buffer.index('{')
                end_idx = buffer.index('}') + 1
                data_str = buffer[start_idx:end_idx]

                # Process the received JSON object
                data_dict = json.loads(data_str)
                print(f'Server Response: {data_dict}')

                # Remove the processed JSON object from the buffer
                buffer = buffer[end_idx:]
        except json.JSONDecodeError as e:
            # Incomplete JSON, continue receiving but log the error
            print(e)
            continue

"""
    @brief: Main function for running the client.
    @param: ip - IP address of the target machine (default: localhost)
            port - Port number for the server (default: 12345)
    @return: None
    @pre: The server is actively listening and reachable
    @post: Runs the client, connecting to the server and receiving data
"""
def main(ip, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            print(f'Connecting to {ip}:{port}...')
            client_socket.connect((ip, port))
            receive_data(client_socket)
    except Exception as e:
        print(f"Something went wrong: {e}")

"""
    @brief: Due to the GIL, this block is placed at the bottom
            to prevent accidental misuse of the script.
    @brief: Executed block when ran from CLI
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='client.py',
        description='TCP Client for receiving data from a listening port on the server',
        epilog='Silas Rodriguez, R11679913, CMPE Senior, 2023'
    )
    parser.add_argument('-i', '--ip', help='IP target machine (default localhost)', type=str, default='localhost')
    parser.add_argument('-p', '--port', help='Port number (default: 12345)', type=int, default=12345)

    argv = parser.parse_args()
    main(argv.ip, argv.port)
    exit(0)

"""
    @sources:

        sockets, select + socket methods:
            https://docs.python.org/3/library/socket.html

        JSON + serialization:
            https://docs.python.org/3/library/json.html
        
        datetime:
            https://docs.python.org/3/library/datetime.html

    @note: I stuck to built-in packages and file handling in order to preserve extendability across machines
"""