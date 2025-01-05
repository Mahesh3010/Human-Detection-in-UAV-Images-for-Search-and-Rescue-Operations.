import cv2
import os

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
    detected_path = os.path.join(detected_folder, 'detected_' + filename)
    cv2.imwrite(detected_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

input_folder = r'F:\heridal\trainImages'
detected_folder = r'F:\heridal\DETECTED_FOLDER_IMAGES'
count = 1
if not os.path.exists(detected_folder):
    os.makedirs(detected_folder)
for filename in os.listdir(input_folder):
    if filename.endswith(('.JPG', '.jpeg', '.png','.jpg')):  
        input_path = os.path.join(input_folder, filename)
        detect_objects(input_path)
        print(count)
        count += 1