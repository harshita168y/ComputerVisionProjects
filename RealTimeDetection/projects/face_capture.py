import time
import cv2
import imutils
import pickle
import statistics
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Load the known faces and embeddings along with OpenCV's Haar cascade for face detection
encodingsP = "encodings.pickle"
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize the video stream and allow the camera sensor to warm up
vs = VideoStream(src=0).start()  # Use standard webcam, remove usePiCamera for compatibility
time.sleep(2.0)  # Sleep to allow camera to warm up

# Get the frame size once, to avoid mismatch during video writing
frame = vs.read()
frame = imutils.resize(frame, width=500)  # Resize for better performance
frame_height, frame_width = frame.shape[:2]

# Initialize video writer to save the video
frame_rate = 1  # Set desired frame rate for the output video (frames per second)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, frame_rate, (frame_width, frame_height))

# Start the FPS counter
fps = FPS().start()

# Variables to track people and objects over time
people_in_scene = {}
confidence_over_time = {}
time_intervals = {}
start_time = time.time()
duration = 20  # Set duration to 200 seconds for this test
total_frames = int(frame_rate * duration)  # Total frames to write based on duration

# Loop over frames from the video file stream
frame_count = 0  # Count the number of frames captured

while frame_count < total_frames:
    # Capture the current timestamp
    current_time = time.time() - start_time

    # Grab the frame from the threaded video stream and resize it to 500px (to speed up processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    frame_height, frame_width = frame.shape[:2]

    # Detect the face boxes
    boxes = face_recognition.face_locations(frame)
    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []
    proba = {}

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"  # If face is not recognized, then print Unknown

        # Check to see if we have found a match
        if True in matches:
            # Find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the recognized face with the largest number of votes
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            fd_mean = statistics.mean([fd if (fd < .5) else 1 for fd in face_distances])
            proba[name] = (1 - fd_mean) * 100
            name = max(counts, key=counts.get)

            # If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                currentname = name

        # Update the list of names
        names.append(name)

        # Track people entering and leaving the scene
        if name != "Unknown":
            if name not in people_in_scene:
                people_in_scene[name] = []
                time_intervals[name] = []
            people_in_scene[name].append(current_time)
            confidence_over_time[name] = proba[name]

    # Loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        text = "{}:{:.2f}%".format(name, proba.get(name, 0))
        # Draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 2)

    # Write the annotated frame to the video file
    out.write(frame)

    # Display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)

    # Introduce a delay to match the target frame rate
    time.sleep(1 / frame_rate)  # Control the frame capture speed

    # Increment the frame count
    frame_count += 1

    # Quit when 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Release the video writer
out.release()

# Plotting the data collected over time
for name, times in people_in_scene.items():
    plt.plot(times, [confidence_over_time[name]] * len(times), label=name)
plt.xlabel('Time (s)')
plt.ylabel('Confidence (%)')
plt.title('Confidence Over Time for Recognized People')
plt.legend()
plt.show()

# Report on detection quality
for name, confidence in confidence_over_time.items():
    if confidence < 50:
        print(f"[WARNING] Low confidence for {name}: {confidence:.2f}%. Lighting or camera focus might be an issue.")

# Cleanup
cv2.destroyAllWindows()
vs.stop()
