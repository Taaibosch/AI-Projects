from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
import numpy as np
from flask_socketio import SocketIO
import os
import random
import threading
import tensorflow as tf
import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)
socketio = SocketIO(app)

# Load the previously saved model
model = tf.keras.models.load_model('Sesotho recognition Model.keras')

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Set parameters
image_size = 100  # The size to which images were resized during training
offset = 20       # Offset for cropping the hand region

# Video capture from webcam
video_cap = cv2.VideoCapture(0)

# Initialize an empty string to store the constructed word
constructed_word = ""
detected_gestures = []  # To store all detected gestures

# Define gesture mapping for letters
gesture_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 
    19: 'U', 20: 'Y', 21: 'KEA U RAPELA', 22: 'KEA U RATA', 23: 'SPACE', 24: 'HO BAPALA', 25: 'HO LOKILE'
}

# Timer for controlling gesture detection intervals
last_detection_time = time.time()
detection_interval = 5  # 5 seconds between detections

def generate_frames():
    global constructed_word, last_detection_time, detected_gestures

    while True:
        success, image = video_cap.read()
        if not success:
            break

        hands, image = detector.findHands(image)

        current_time = time.time()
        if hands and (current_time - last_detection_time >= detection_interval):
            last_detection_time = current_time

            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Prepare white background for hand image
            image_white = np.ones((image_size, image_size, 3), np.uint8) * 255
            image_crop = image[y - offset: y + h + offset, x - offset: x + w + offset]

            if image_crop.size == 0:
                continue  # Skip if crop is empty

            aspect_ratio = h / w
            if aspect_ratio > 1:
                constant = image_size / h
                width_calculated = int(constant * w)
                image_resize = cv2.resize(image_crop, (width_calculated, image_size))
                width_gap = (image_size - width_calculated) // 2
                image_white[:, width_gap: width_gap + width_calculated] = image_resize
            else:
                constant = image_size / w
                height_calculated = int(constant * h)
                image_resize = cv2.resize(image_crop, (image_size, height_calculated))
                height_gap = (image_size - height_calculated) // 2
                image_white[height_gap: height_gap + height_calculated, :] = image_resize

            image_white_normalized = image_white / 255.0
            image_input = np.expand_dims(image_white_normalized, axis=0)

            # passing normalized image to the model to return the prediction
            prediction = model.predict(image_input)
            predicted_class = np.argmax(prediction, axis=1)[0]
            print(prediction)

            predicted_gesture = gesture_mapping.get(predicted_class, "Unknown")

            # Making space between text
            if predicted_gesture == 'SPACE':
                predicted_gesture = " "
                constructed_word += predicted_gesture
                detected_gestures.append(predicted_gesture)
                print("found SPACE and making space")
                print(detected_gestures)

            constructed_word += predicted_gesture
            detected_gestures.append(predicted_gesture)  # Append the gesture to the list

        ret, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
        
    # print(detected_gestures)

def get_random_gesture(letter):

    if os.path.exists(f'Data_collected/{letter}'):
        print('path found')
    gesture_folder = f'Data_collected/SPACE' if letter == " " else f'Data_collected/{letter}'
    
    if os.path.exists(gesture_folder):
        gesture_images = [f for f in os.listdir(gesture_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if gesture_images:
            selected_image = random.choice(gesture_images)
            return f"/get_image/{letter}/{selected_image}"
    return None


def handle_gesture_display(word):
    global constructed_word
    main_folder = f'Data_collected/{word}'
    if os.path.exists(main_folder):
        gesture_path = get_random_gesture(word)
        # constructed_word += letter
        print('The path found')
        socketio.emit('new_gesture', {'path': gesture_path})  # Emit the gesture
        socketio.sleep(2)  # Wait for 2 seconds before emitting the next gesture
    else:
        print('The path not found')
        for letter in word:
            gesture_path = get_random_gesture(letter)
            if gesture_path:
                constructed_word += letter
                socketio.emit('new_gesture', {'path': gesture_path})  # Emit the gesture
                socketio.sleep(2)  # Wait for 2 seconds before emitting the next gesture


@app.route('/get_image/<letter>/<filename>')
def get_image(letter, filename):
    # gesture_folder = f'Data_Collected/SPACE' if letter == " " else f'Data_Collected/{letter}'

    if letter == " ":
        gesture_folder = f'Data_Collected/SPACE'
    else:
        gesture_folder = f'Data_Collected/{letter}'

    return send_from_directory(gesture_folder, filename)

@socketio.on('process_text')
def process_text(data):
    user_input = data['text_input']
    threading.Thread(target=handle_gesture_display, args=(user_input,)).start()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/get_constructed_word')
# def get_constructed_word():
#     global detected_gestures
#     gestures_to_send = detected_gestures.copy()  # Send a copy of the current gestures

#     # update_constructed_word()

#     # print(gestures_to_send)
#     return jsonify({'word': constructed_word, 'gestures': gestures_to_send})

previous_word = "" 

@app.route('/get_constructed_word')
def get_constructed_word():
    global detected_gestures, previous_word
    
    current_word = ''.join(detected_gestures)

    is_deleted = False

    if len(current_word) < len(previous_word):
        is_deleted = True

    
    if len(current_word) < len(previous_word):
        # Send updated information to the second function
        update_constructed_word(current_word)

    previous_word = current_word

    gestures_to_send = detected_gestures  # Send a copy of the current gestures
    return jsonify({"gestures": gestures_to_send, "is_deleted": is_deleted}), 200

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('comBridge.html')  # Render your HTML page

@app.route('/update_constructed_word', methods=['POST'])
def update_constructed_word():
    data = request.json
    updated_word = data.get('updated_word')
    if updated_word:
        print(updated_word, type(updated_word))
        return jsonify({"gestures": updated_word}), 200
    else:
        return jsonify({"error": "No word received"}), 400


if __name__ == '__main__':
    # app.run()
    app.run(debug=True, host='0.0.0.0', port=5000) 



