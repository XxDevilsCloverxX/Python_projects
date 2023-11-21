"""
    @author: Silas Rodriguez
    @program: server.py
    @brief: TCP Server for receiving data on a listening port,
            acts like an API & transmits using JSON serialization
    @date: 11/20/2023
"""

import socket   # Used for creating and managing sockets
import argparse # Used for parsing command line arguments
from select import select  # Used for I/O multiplexing - keyboard intterupt
from datetime import datetime   # Used for working with timestamps
import json     # Used for JSON encoding/decoding - note at the bottom

"""
    @brief: Processes a data line from the input file into a dictionary.
    @param: line - List containing values of a data line
    @return: A dictionary containing processed data
    @pre: The line is properly formatted with required data
    @post: Returns a dictionary representing the processed data
"""
def process_data_line(line: list):
    # Process the ZTIME column (assuming it's a timestamp)
    timestamp = datetime.strptime(line[0], "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

    # Construct a dictionary for the data, including timestamp
    data_dict = {
        'timestamp': timestamp,
        'LON': float(line[1]),
        'LAT': float(line[2]),
        'WSR_ID': line[3],
        'CELL_ID': line[4],
        'RANGE': int(line[5]),
        'AZIMUTH': int(line[6]),
        'SEVPROB': int(line[7]),
        'PROB': int(line[8]),
        'MAXSIZE': float(line[9])
    }
    return data_dict

"""
    @brief: Sends processed data to the connected client socket.
    @param: client_socket - Socket object for the connected client
            file_path - Path to the input file
    @return: None
    @pre: The client socket is connected and the file exists
    @post: Sends processed data to the client socket
"""
def send_data(client_socket: socket.socket, file_path: str):
    # open the file specified in file path and close when done
    with open(file_path, 'r') as file:
        # using a generator, take a line at a time
        for i, line in enumerate(file.readlines()):
            if line.startswith('#'):  # Skip the comments line
                continue
            if i%1000 == 0:
                print(f'Transmitting line {i+1}...')

            # Split the line into individual values
            line_values = line.strip().split(',')

            # Process and send data lines to the client
            data_dict = process_data_line(line_values)
            client_socket.send(json.dumps(data_dict).encode())
        print('Transmission completed successfully!')
"""
    @brief: Main function for running the server.
    @param: file_path - Path to the input file
            port - Port number for the server (default: 12345)
    @return: None
    @pre: The input file exists, and the port is available
    @post: Runs the server, listening for incoming connections and sending data to clients
"""
def main(file_path: str, port: int):
    # create a server socket that closes appropriately when the with statement exits
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('', port))  # bind all network interfaces on host to this port (demux ip+port -> this TCP conn)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # releases socket after use to immediately be reused
        server_socket.listen(1) # make this a listening socket

        #acquire this server's IP address for ease of use testing on local host
        host_ip = socket.gethostbyname(socket.gethostname())
        print(f"Server listening on {host_ip}:{port}")

        # This try-except block allows for most errors to keep the server alive,
        # but Ctrl+C is used to exit the server gracefully.
        try:
            while True:
                readable, _, _ = select([server_socket], [], [], 1)
                # when its the server_sockets turn to act on a connection (every 1s)
                if server_socket in readable:
                    # this try block allows for the server to experience connection interruptions and failures without going offline
                    try:
                        client_socket, client_address = server_socket.accept()
                        print(f"Connection from {client_address}!")

                        # do work with the new connection and close it appropriately
                        with client_socket:
                            send_data(client_socket, file_path)
                    # log what happened with the connection failure + continue
                    except Exception as e:
                        print(f'Error occurred between client: {client_address} and server.')
                        print(f'Connection closed: {e}')
        # Exit gracefully from the server side - closes socket
        except KeyboardInterrupt:
            print("Server interrupted. Closing...")

"""
    @brief: Due to the GIL, this block is placed at the bottom
            to prevent accidental misuse of the script.
    @brief: Executed block when ran from CLI
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='server.py',
        description='TCP Server for receiving data on a listening port',
        epilog='Silas Rodriguez, R11679913, CMPE Senior, 2023'
    )
    parser.add_argument('-f', '--file', help='File path', required=True, type=str)
    parser.add_argument('-p', '--port', help='Port number (default: 12345)', required=False, type=int, default=12345)

    argv = parser.parse_args()
    main(argv.file, argv.port)
    exit(0)

"""
Perks of using JSON serialization on this application:
    Human-Readable and Easy to Debug:
        JSON is a text-based format that is easy for humans to read and understand. This makes it convenient for debugging and manual inspection of the transmitted data.

    Interoperability:
        JSON is a widely supported data interchange format. It is language-agnostic, meaning it can be easily consumed by applications written in different programming languages. This makes it suitable for creating an API-like communication between the server and the client.

    Serialization and Deserialization:
        JSON encoding and decoding, also known as serialization and deserialization,
        are built-in features in many programming languages. Complex data structures like Python dictionaries
        are easliy converted to a format that can be transmitted over the network and then reconstructed.
        - Other option is Pickle, but is less - well known for networking and more Pythonic.

    Standardized Format:
        JSON provides a standardized format for representing key-value pairs and structured data.
        This standardization simplifies the communication process between the server and client,
        as both sides can rely on a common format.

    API Usage:
        JSON is commonly used in web APIs for data exchange between servers and clients.
        By utilizing JSON in this server-client application, align with common practices
        and industry standards, making it easier to integrate with other systems or APIs

    Lightweight and Efficient:
        JSON is a lightweight data interchange format, and its efficient
        encoding and decoding processes contribute to faster data transmission over the network.
"""

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

