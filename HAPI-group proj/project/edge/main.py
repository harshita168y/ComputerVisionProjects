import sensor, time
import uasyncio as asyncio
import gc


# import for project functions
from support_functions import comms_setup, channels_setup, sendCommand, sendData, init_sensor, load_model_and_labels, init_params, load_config, debug_print, send_inference_data, read_battery_voltage

# parameters initialisation
params = init_params()
confidence_threshold = params["confidence_threshold"]
gesture_detection_blackout_msecs = params["gesture_detection_blackout_msecs"]
last_command_time = params["last_command_time"]

config = load_config() # Load comms config from a config.json file

# comms setup
comms_setup(config["SSID"], config["PASSWORD"]) # Connect using WiFi credentials

# channels setup
channels_setup(config["SERVER_IP"], config["COMMANDS_PORT"], config["DATA_PORT"])

# sensor initialisation
init_sensor(config["WINDOW_SIZE"])

# load model
net, labels = load_model_and_labels("trained.tflite", "labels.txt")
time.sleep(1)

clock = time.clock()


async def gesture_command_task():
    global last_command_time

    # some locals
    gesture_buffer = None
    gesture_buffer_count = 0
    gesture_required_count = 3  # How many frames required before sending

    while True:
        # capture image
        clock.tick()
        img = sensor.snapshot()

        # Measure inference time
        predictions, inference_time_ms = do_inference(img)
        await asyncio.sleep_ms(1)  # let event loop breathe


        # determine probabilities, only send if above a confidence threshold and if not throttling
        predictions_list = list(zip(labels, predictions))

        now = time.ticks_ms()

        # Check if cooldown expired
        if time.ticks_diff(now, last_command_time) > gesture_detection_blackout_msecs:
            detected_gesture = None
            highest_score = 0

            # Find highest scoring gesture above threshold
            for label, score in predictions_list:
                if label != "Unknown" and score > confidence_threshold:
                    if score > highest_score:
                        detected_gesture = label
                        highest_score = score

            if detected_gesture:
                if gesture_buffer == detected_gesture:
                    gesture_buffer_count += 1
                else:
                    gesture_buffer = detected_gesture
                    gesture_buffer_count = 1

                # Check if gesture is stable enough
                if gesture_buffer_count >= gesture_required_count:
                    debug_print(f"Confirmed gesture: {detected_gesture} ({highest_score:.2f})")
                    sendCommand(detected_gesture)
                    last_command_time = now
                    gesture_buffer = None
                    gesture_buffer_count = 0
            else:
                # No valid gesture detected
                gesture_buffer = None
                gesture_buffer_count = 0


        # let other things get a chance
        await asyncio.sleep_ms(1)  # Reduced sleep time

async def telemetry_data_task():
    while True:
        gc.collect() # periodic memory cleaning - identifies if we have leaks!

        # Send telemetry data periodically with synchronization
        telemetry_data = f"Mem: {gc.mem_free()}, Batt: {read_battery_voltage()}"
        sendData(telemetry_data)  # this is a fire and forget, non blocking

        # let other things get a chance
        await asyncio.sleep(30)  # Send telemetry data every 30 seconds

def do_inference(img):
    t_start = time.ticks_us()
    predictions = net.predict([img])[0].flatten().tolist()
    t_end = time.ticks_us()
    inference_time_ms = time.ticks_diff(t_end, t_start) / 1000  # convert to ms
    return predictions, inference_time_ms

async def main():
    await asyncio.gather(
        gesture_command_task(),
        telemetry_data_task()
    )

if __name__ == "__main__":
    gc.collect()
    asyncio.run(main())
