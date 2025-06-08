
# HAPI: Hand-gesture-based Action interface, Personalized for the Individual

HAPI is an Edge AI proof-of-concept system that empowers individuals with limited mobility or communication ability to perform predefined actions using simple hand gestures. It is optimized for low-power embedded devices, demonstrating real-time gesture recognition and action mapping via compact hardware and lightweight models.

## Project Summary

- **Device**: OpenMV Cam RT1062 (ARM Cortex-M7 @ 600 MHz)
- **Model**: Transfer-learned MobileNetV2 via TensorFlow/Keras
- **Recognition**: 2 gestures + background (Thumbs Up, Thumbs Down, Unknown)
- **Communication**: Wi-Fi + UDP between edge, client, and server
- **Features**:
  - Gesture-to-action mapping
  - Text-to-speech via gTTS
  - Real-time telemetry (battery, memory)
  - Edge device runs MicroPython

## Repository Structure

```
/server          → Python server with gesture-action dispatch + gTTS
/client          → Python GUI to configure gesture-action mappings (Tkinter)
/edge            → MicroPython app for OpenMV Cam RT1062 (main.py, functions.py)
/common          → Shared label IDs and configuration files
/tf_keras        → Model training, evaluation, grid search scripts
```

## Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/DylanAtUL/dle-project.git
cd dle-project
```

### 2. Launch Applications

Open three terminals:

- **Server**
  ```bash
  cd server
  python hapi_server.py
  ```

- **Client**
  ```bash
  cd client
  python hapi_client.py
  ```

- **Edge**
  Flash `edge/main.py` and `supporting_functions.py` to the OpenMV Cam RT1062 using OpenMV IDE.

### 3. Configure Gestures

Use the client GUI to map gestures (e.g., “Thumbs Up”) to text commands like:
```
"Alexa, turn on the lights."
```

## Model Details

- Base: MobileNetV2 (α=0.1), ImageNet pre-trained
- Input Size: 96x96 RGB
- Final Output: 3-class SoftMax
- Accuracy: ~94% (Thumbs Up / Down / Unknown)
- Size: <300KB (optimized for 512KB heap)

## Grid Search Hyperparameters

```python
'PIXEL_WIDTH': [96, 128, 192]
'LEARNING_RATE': [0.001, 0.005, 0.01]
'ALPHA': [0.1]
'BATCH_SIZE': [16, 32]
'EPOCHS': [20]
'FINAL_SPARSITY': [0.5, 0.7, 0.8]
```

## Requirements

- Python 3.8+
- OpenMV IDE (for flashing edge firmware)
- Dependencies:
  ```bash
  pip install gtts
  sudo apt-get install mpg123
  ```

## Logs

Each app prints logs to terminal:
- Server: Gesture ID, mapped speech, telemetry
- Client: Config save confirmation
- Edge: Heap usage, battery, gesture inference

## Hardware Notes

- OpenMV Cam RT1062
  - Heap: 512KB
  - Flash: 16MB total, 4MB usable
  - FrameBuffer: 32MB
- gTTS is used for on-laptop speech output
- Battery and buzzer support partially implemented (requires breadboard mods)

## References

- Report: See `temp.docx` for full system overview and results
- OpenMV Docs: https://openmv.io/
- Edge Impulse: https://studio.edgeimpulse.com
- gTTS: https://pypi.org/project/gTTS/

## Contributors

Group 5 – University of Limerick, Module:EE6008-DEEP LEARNING AT THE EDGE 2024/5 SEM2, Project:Edge AI Application Using OpenMV Cam RT1062

---


