import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import pygame
from plyer import notification

model = YOLO('best.pt')

pygame.init()
alarm_sound = pygame.mixer.Sound('alarm_sound.mp3')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


def play_alarm_sound():
    # Play an alarm sound when an accident is detected
    alarm_sound.play()


def send_notification(message):
    # Send a notification when an accident is detected
    notification.notify(
        title='Accident Detected!',
        message=message,
        app_icon=None,
        timeout=10
    )


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cr.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        # Check if the detected object is an accident (you may need to adjust this condition based on your detection results)
        if c == 'Accident':
            play_alarm_sound()
            print('Accident Detected!')
            send_notification("An accident has been detected!")

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
