import sensor, time
import uasyncio as asyncio
import gc


# import for project functions
from support_functions import comms_setup, channels_setup, sendCommand, sendData, init_sensor, load_model_and_labels, init_params, load_config, debug_print, send_inference_data, initialise_time, read_battery_voltage


# parameters initialisation
params = init_params()
confidence_threshold = params["confidence_threshold"]
command_cooldown_ms = params["command_cooldown_ms"]
last_command_time = params["last_command_time"]

config = load_config() # Load comms config from a config.json file

# comms setup
comms_setup(config["SSID"], config["PASSWORD"]) # Connect using WiFi credentials

# channels setup
channels_setup(config["SERVER_IP"], config["COMMANDS_PORT"], config["DATA_PORT"])

# sensor initialisation
init_sensor()

# load model
net, labels = load_model_and_labels("trained.tflite", "labels.txt")
time.sleep(1)

# timing init
initialise_time()
clock = time.clock()


async def gesture_command_task():
    global last_command_time
    while True:
        # capture image
        clock.tick()
        img = sensor.snapshot()

        # Measure inference time
        predictions, inference_time_ms = do_inference(img)

        # determine probabilities, only send if above a confidence threshold and if not throttling
        predictions_list = list(zip(labels, predictions))
        for label, score in predictions_list:
            if label != "Unknown" and score > confidence_threshold:
                now = time.ticks_ms()
                if time.ticks_diff(now, last_command_time) > command_cooldown_ms:
                    debug_print(f"Sending command for: {label} ({score:.2f})")
                    sendCommand(f"{label}")  # Ensure this is non-blocking
                    last_command_time = now
                    # Send shared data with synchronization
                    data = f"Inference time: {inference_time_ms:.2f} ms, FPS: {clock.fps()}"
                    send_inference_data(data)  # Ensure this is non-blocking

        # let other things get a chance
        await asyncio.sleep_ms(10)  # Reduced sleep time

async def telemetry_data_task():
    while True:
        gc.collect() # periodic memory cleaning - identifies if we have leaks!

        # Send telemetry data periodically with synchronization
        telemetry_data = f"Mem: {gc.mem_free()}, Batt: {read_battery_voltage()}"
        debug_print(telemetry_data)
        sendData(telemetry_data)  # this is a fire and forget, non blocking

        # let other things get a chance
        await asyncio.sleep(10)  # Send telemetry data every 30 seconds

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
