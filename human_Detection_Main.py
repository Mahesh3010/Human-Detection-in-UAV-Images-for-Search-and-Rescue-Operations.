from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER_IMAGES = 'UPLOAD_FOLDER_IMAGES'
DETECTED_FOLDER_IMAGES = 'DETECTED_FOLDER_IMAGES'
RESIZED_FOLDER = 'Resized_Images'

app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES
app.config['DETECTED_FOLDER_IMAGES'] = DETECTED_FOLDER_IMAGES
app.config['RESIZED_FOLDER'] = RESIZED_FOLDER

def Resize_Image(image_path):
    efficientnet = tf.keras.applications.EfficientNetB0(weights='imagenet')
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))
    resized_image = tf.keras.applications.efficientnet.preprocess_input(resized_image)
    input_image = tf.expand_dims(resized_image, axis=0)
    predictions = efficientnet.predict(input_image)
    detect_objects(image_path)

def detect_objects(image_path):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (rects, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    count = 0
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 4)
        count += 1
    final_image = cv2.putText(image, f"Number of humans detected: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
    detected_path = os.path.join('C:\\Users\\ue\\3D Objects\\Final Year Project Folder\\Final FTR Project\\New\\Final Year Project Folder\\Implementation', app.config['DETECTED_FOLDER_IMAGES'], 'detected_' + os.path.basename(image_path))
    cv2.imwrite(detected_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    return detected_path

def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join('C:\\Users\\ue\\3D Objects\\Final Year Project Folder\\Final FTR Project\\New\\Final Year Project Folder\\Implementation', app.config['DETECTED_FOLDER_VIDEOS'], 'detected_' + os.path.basename(input_video_path))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_frame = detect_objects(frame)
        out.write(detected_frame)
    cap.release()
    out.release()
    return output_video_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            if filename.endswith(('.jpg', '.jpeg', '.png','.JPG')):
                upload_path = os.path.join('C:\\Users\\ue\\3D Objects\\Final Year Project Folder\\Final FTR Project\\New\\Final Year Project Folder\\Implementation', app.config['UPLOAD_FOLDER_IMAGES'], filename)
                file.save(upload_path)
                detect_objects(upload_path)
                # Resize_Image(upload_path)

            elif filename.endswith('.mp4'):
                upload_path = os.path.join('C:\\Users\\ue\\3D Objects\\Final Year Project Folder\\Final FTR Project\\New\\Final Year Project Folder\\Implementation', app.config['UPLOAD_FOLDER_VIDEOS'], filename)
                file.save(upload_path)
                process_video(upload_path)
            return redirect(url_for('index'))
    return redirect(url_for('index'))

# @app.route('/results/<filename>')
# def show_results(filename):
#     resized_path = os.path.join(app.config['RESIZED_FOLDER'], 'resized_' + filename)
#     detected_path = os.path.join(app.config['DETECTED_FOLDER_VIDEOS'], 'detected_' + filename)
#     return render_template('results.html', resized_path=resized_path, detected_path=detected_path)

if __name__ == '__main__':
    app.run(debug=True)