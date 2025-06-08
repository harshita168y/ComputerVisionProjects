import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter
from matplotlib.patches import Patch

# Function: Encode Faces
def encode_faces(dataset_path="dataset/"):
    known_encodings = []
    known_names = []
    for name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, name)
        if os.path.isdir(person_path):
          for file in os.listdir(person_path):
            image_path = os.path.join(person_path, file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
              known_encodings.append(encodings[0])
              known_names.append(name)
              
    with open("encodings.pickle", "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print("Encodings saved successfully!")

# Function: Load TFLite Model
def load_tflite_model(model_path, label_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load labels
    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return interpreter, input_details, output_details, labels

# Function: Perform TFLite Object Detection
def detect_objects_tflite(frame, interpreter, input_details, output_details, labels, conf_threshold=0.6):
    height, width, _ = frame.shape
    input_shape = input_details[0]['shape'][1:3]

    # Preprocess frame
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    detected_objects = []
    for i in range(len(scores)):
        if scores[i] > conf_threshold:
            ymin, xmin, ymax, xmax = boxes[i]

            # Convert normalized coordinates to pixel values
            x = int(xmin * width)
            y = int(ymin * height)
            w = int((xmax - xmin) * width)
            h = int((ymax - ymin) * height)

            # Ensure the bounding box remains within the image bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))

            class_id = int(classes[i])
            detected_objects.append((labels[class_id], scores[i], x, y, w, h))

    return detected_objects

# Function: Check if Image is Blurry
def is_image_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

# Function: Log Detection Issues
def log_detection_issues(confidence, box_size, image_blur):
    if confidence < 0.5:
        print(f"Warning: Low detection confidence: {confidence}")
    if box_size < 0.05 or box_size > 0.5:
        print(f"Warning: Bounding box size is abnormal: {box_size}")
    if image_blur:
        print("Warning: The image seems blurry. Try improving the focus or lighting.")

# Function: Real-Time Detection
# Function: Real-Time Detection with Improved "Unknown" Detection
def real_time_detection(duration=20, skip_frames=10, output_video_path="output_video.avi"):
    # Load face encodings
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)

    # Load TFLite model
    model_path = "ssd_mobilenet_v2.tflite"
    label_path = "labelmap.txt"
    interpreter, input_details, output_details, labels = load_tflite_model(model_path, label_path)

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    detections_log = []
    frame_count = 0

    # Initialize VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 3.0, (640, 480))  # Adjust frame size if needed

    # For plotting data (e.g., confidence values over time)
    object_detections = []

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Optimize frame size
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Detect objects using TFLite
        objects = detect_objects_tflite(small_frame, interpreter, input_details, output_details, labels)

        # Check for image quality (blurriness)
        image_blur = is_image_blurry(frame)
        
        # Process faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"
            confidence = 0
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            best_match_distance = face_distances[best_match_index]

            # Set a threshold to label as "Unknown" if the match confidence is low
            if best_match_distance < 0.5:  # Adjust this threshold as necessary
                name = data["names"][best_match_index]
                confidence = 1 - best_match_distance  # The confidence is 1 minus the distance
            else:
                name = "Unknown"
                confidence = 0

            # Log detections
            detections_log.append({
                "time": time.time() - start_time,
                "type": "person",
                "name": name,
                "confidence": confidence
            })

            # Draw bounding box for face
            left, top, right, bottom = left * 2, top * 2, right * 2, bottom * 2  # Scale back to original size
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process objects
        for obj in objects:
            label, conf, x, y, w, h = obj

            # Avoid showing unknown person or label with low confidence
            if label != "unknown person" and conf > 0.4:
                # Log detections
                detections_log.append({
                    "time": time.time() - start_time,
                    "type": "object",
                    "name": label,
                    "confidence": conf
                })

                # Store detection data for plotting
                object_detections.append((time.time() - start_time, label, conf))

                # Check bounding box size
                box_size = (w * h) / (frame.shape[0] * frame.shape[1])  # Calculate box size as a fraction of the image
                log_detection_issues(conf, box_size, image_blur)

                # Draw bounding box for object with confidence
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the frame to video file
        out.write(frame)

        # Display frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()

    return detections_log

def plot_detection_log(detections_log):
    times = [log["time"] for log in detections_log]
    unique_names = list({log["name"] for log in detections_log})  # Unique names for the legend

    # Initialize plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_names)))
    name_to_color = dict(zip(unique_names, colors))

    for log in detections_log:
        name = log["name"]
        confidence = log["confidence"]
        time = log["time"]

        # Plot interval as a scatter point
        plt.scatter(time, confidence, label=name, color=name_to_color[name], alpha=0.8)

    # Add legend and labels
    patches = [Patch(color=name_to_color[name], label=name) for name in unique_names]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Detection Timeline with Confidence Levels")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.grid(True)
    plt.show()

# Main Execution
if __name__ == "__main__":
    detections_log = real_time_detection(duration=50, skip_frames=5)
    # Step 3: Plot the Results
    plot_detection_log(detections_log)
