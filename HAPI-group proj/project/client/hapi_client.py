import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json
import os
import socket
from datetime import datetime

spinner_running = False
spinner_index = 0
spinner_frames = ["Saving.", "Saving..", "Saving..."]

# used when the user clicks the save button
def update_spinner():
    global spinner_index
    if spinner_running:
        status_label.config(text=spinner_frames[spinner_index % len(spinner_frames)])
        spinner_index += 1
        root.after(500, update_spinner)


# ------------------ CONFIG LOAD ------------------

CONFIG_PATH = "config.json"
USER_GESTURE_DEFINITIONS = "user_config.json"  # Stores user-entered labels
GESTURE_IDS = "../common/gesture_ids.json"                    # Stores valid gesture identifiers

try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
except Exception as e:
    print(f"Failed to load {CONFIG_PATH}: {e}")
    exit(1)

try:
    with open(GESTURE_IDS, "r") as f:
        GESTURES = json.load(f)
except Exception as e:
    print(f"Failed to load {GESTURE_IDS}: {e}")
    exit(1)

# Extracted config values
DEBUG = CONFIG.get("debug", True)
IP = CONFIG.get("ip")
CLIENT_PORT = CONFIG.get("client_port")


# ------------------ DEBUG LOGGER ------------------

def debug_print(msg):
    if DEBUG:
        print(f"{msg}")


# ------------------ APP LOGIC ------------------

entries = {}

# Load previously saved user gesture mappings
try:
    with open(USER_GESTURE_DEFINITIONS, "r") as f:
        user_config = json.load(f)
        debug_print(f"Loaded user config: {user_config}")
except Exception:
    user_config = {}
    debug_print("No existing user config found or failed to load.")


def save_config():
    global spinner_running

    # Start spinner
    spinner_running = True
    spinner_index = 0
    update_spinner()
    save_btn.config(state=tk.DISABLED)
    exit_btn.config(state=tk.DISABLED)

    # Start save in a background thread so UI doesn't freeze
    def do_save():
        global spinner_running
        try:
            config = {}
            for gesture_id, entry in entries.items():
                text = entry.get().strip()
                config[gesture_id] = text

            debug_print(f"Collected gesture config: {config}")

            with open(USER_GESTURE_DEFINITIONS, "w") as f:
                json.dump(config, f, indent=4)
            debug_print(f"Saved to {USER_GESTURE_DEFINITIONS}")

            with open(USER_GESTURE_DEFINITIONS, "r") as f:
                gesture_data = f.read()

            debug_print(f"Sending gesture config to {IP}:{CLIENT_PORT}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                sock.connect((IP, CLIENT_PORT))
                sock.sendall(gesture_data.encode("utf-8"))
                response = sock.recv(1024).decode("utf-8")
                debug_print(f"Received response: {response}")

            if "OK" in response.upper():
                messagebox.showinfo("Success", "Gesture config sent successfully.")
            # TODO else:
            #    messagebox.showwarning("Warning", f"Unexpected response: {response}")

        except Exception as e:
            messagebox.showerror("Send Error", f"Failed to save/send config: {e}")
            debug_print(f"Send failed: {e}")
        finally:
            spinner_running = False
            save_btn.config(state=tk.NORMAL)
            exit_btn.config(state=tk.NORMAL)
            status_label.config(text="")  # Clear spinner

    import threading
    threading.Thread(target=do_save, daemon=True).start()

# ------------------ GUI SETUP ------------------

root = tk.Tk()
root.title("Gesture Command Setup")
root.geometry("900x520")
root.resizable(False, False)

title = tk.Label(root, text="Hapi Client Setup", font=("Arial", 16, "bold"))
title.pack(pady=10)

top_frame = tk.Frame(root)
top_frame.pack()

# Gesture config frame
config_frame = tk.Frame(top_frame)
config_frame.pack(padx=20, pady=10)

for gesture_id in GESTURES:
    row = tk.Frame(config_frame)
    row.pack(pady=5)

    icon_path = os.path.join("gesture_icons", f"{gesture_id}.png")
    if os.path.exists(icon_path):
        img = Image.open(icon_path).resize((50, 50))
        icon = ImageTk.PhotoImage(img)
        icon_label = tk.Label(row, image=icon)
        icon_label.image = icon
    else:
        icon_label = tk.Label(row, text=gesture_id.upper(), width=10)

    icon_label.pack(side="left", padx=5)

    entry = tk.Entry(row, width=40)

    # Load initial value if it exists
    if gesture_id in user_config:
        entry.insert(0, user_config[gesture_id])

    entry.pack(side="left", padx=5)
    entries[gesture_id] = entry


# Status label
status_label = tk.Label(root, text="", font=("Arial", 10), fg="blue")
status_label.pack()

# Button frame at the bottom
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

# Save button
save_btn = tk.Button(button_frame, text="Save Configuration", font=("Arial", 12), command=save_config)
save_btn.pack(side=tk.LEFT, padx=10)

# Exit button
exit_btn = tk.Button(button_frame, text="Exit", font=("Arial", 12), command=root.quit)
exit_btn.pack(side=tk.LEFT, padx=10)


root.mainloop()
