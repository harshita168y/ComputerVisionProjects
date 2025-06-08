import socket
from gtts import gTTS
import json
import os
import threading

# ------------------ FILE PATHS ------------------

CONFIG_FILE = "config.json"
GESTURE_IDS_FILE = "../common/gesture_ids.json"
COMMAND_CONFIG_FILE = "gesture_defaults.json"
RESPONSE_FILE = "response.mp3"

# ------------------ CONFIG LOADING ------------------

def load_config(path=CONFIG_FILE):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

def load_command_responses(path=COMMAND_CONFIG_FILE):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No existing {COMMAND_CONFIG_FILE} found. Starting with empty command responses.")
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to load command config: {e}")

def save_command_responses(command_responses, path=COMMAND_CONFIG_FILE):
    try:
        with open(path, "w") as f:
            json.dump(command_responses, f, indent=4)
            print(f"Saved updated gesture commands to {path}")
    except Exception as e:
        print(f"Failed to save updated config: {e}")

# ------------------ SPEAK RESPONSE ------------------

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save(RESPONSE_FILE)
    os.system(f"mpg123 {RESPONSE_FILE}")
    os.remove(RESPONSE_FILE)
# ------------------ HANDLE GESTURE COMMANDS ------------------

def handle_gesture_commands(server_socket, command_responses, gesture_ids):
    palm_status = False
    while True:
        command, client_address = server_socket.recvfrom(1024)
        command = command.decode("utf-8").strip()
        print("Received from:", client_address)

        if not command:
            print("Empty command.")
        else:
            print("Received command:", command)
            response = command_responses.get(command, f"Gesture '{command}' not recognized.")
            print("Speaking:", response)
            speak(response)

# ------------------ HANDLE TELEMETRY ------------------

def handle_telemetry_data(telemetry_socket):
    while True:
        data, addr = telemetry_socket.recvfrom(1024)
        print(f"Received telemetry from {addr}: {data.decode()}")

# ------------------ HANDLE CONFIG UPDATES ------------------

def handle_client_updates(update_socket, command_responses, gesture_ids):
    while True:
        client_socket, client_address = update_socket.accept()
        print("Client connected for config update:", client_address)
        try:
            data = client_socket.recv(4096).decode("utf-8")
            user_config = json.loads(data)

            # Validate keys against gesture_ids
            for gesture in user_config:
                if gesture not in gesture_ids:
                    print(f"Warning: Unknown gesture '{gesture}' in user config.")

            command_responses.clear()
            command_responses.update(user_config)
            print("Gesture command responses updated via socket.")
            print(user_config)

            save_command_responses(command_responses)

        except Exception as e:
            print("Error receiving config update:", e)

        finally:
            client_socket.close()

# ------------------ SERVER SETUP ------------------

def run_server(config):
    host = config["host"]
    port = config["port"]
    data_port = config["data_port"]
    client_port = config["client_port"]

    # Load gesture IDs
    try:
        with open(GESTURE_IDS_FILE, "r") as f:
            gesture_ids = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load {GESTURE_IDS_FILE}: {e}")

    # Load default command mappings
    command_responses = load_command_responses()

    # Gesture socket
    gesture_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gesture_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    gesture_socket.bind((host, port))
    print(f"Listening for gesture commands (UDP) on {host}:{port}")


    # Telemetry socket
    telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    telemetry_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    telemetry_socket.bind((host, data_port))
    print(f"Listening for telemetry data on {host}:{data_port}")

    # Config update socket
    update_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    update_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    update_socket.bind((host, client_port))
    update_socket.listen(1)
    print(f"Listening for user config updates on {host}:{client_port}")

    # Threads
    t1 = threading.Thread(target=handle_gesture_commands, args=(gesture_socket, command_responses, gesture_ids))
    t2 = threading.Thread(target=handle_telemetry_data, args=(telemetry_socket,))
    t3 = threading.Thread(target=handle_client_updates, args=(update_socket, command_responses, gesture_ids))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

# ------------------ MAIN ------------------

if __name__ == "__main__":
    config = load_config()
    run_server(config)
