from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import tkinter as tk



def pop_up_window(detected):

    result = {"action": None}

    popup = tk.Tk()
    popup.title("Item Detected!")
    if (detected.lower() == "cup") or (detected.lower() == "bottle"):
        popup.configure(bg="#63a66f")

        label = tk.Label(
        popup,
        bg="#63a66f",
        text=f"Detected: {detected.lower()}\nThis item is RECYCLABLE!\nGo ahead and throw it in the blue/green bin!",
        fg="white",
        font=("Calibri", 16, "bold"),
        pady=20

        )

    else:
        popup.configure(bg="#8B0000")

        label = tk.Label(
        popup,
        bg="#8B0000",
        text=f"Detected: {detected.lower()}\nThis item is NOT RECYCLABLE!\nGo ahead and throw it in the black bin!",
        fg="white",
        font=("Calibri", 16, "bold"),
        pady=20
        )

    popup.geometry("400x200+500+300")
    popup.resizable(False, False)
    label.pack()

    return buttons(result, popup)

def buttons(result, window):
    buttonFrame = tk.Frame(window, bg="#000000")
    buttonFrame.pack(pady=5)

    def continueFunc():
        result["action"] = "continue"
        window.destroy()

    def exitFunc():
        result["action"] = "exit"
        window.destroy()
        


    continue_btn = tk.Button(
        buttonFrame,
        text="Continue Scanning Items?",
        command= continueFunc,
        bg="#a1aee6",
        fg="white",
        font=("Arial", 12, "bold"),
        padx=15,
        pady=8,
        relief="flat",
        cursor="hand2"
    )
    continue_btn.pack(side="left", padx=10)

    exit_btn = tk.Button(
        buttonFrame,
        text="Exit",
        command=exitFunc,
        bg="#e6a1c2",
        fg="white",
        font=("Arial", 12, "bold"),
        padx=15,
        pady=8,
        relief="flat",
        cursor="hand2"
    )
    exit_btn.pack(side="left", padx=10)

    window.mainloop()
    return result["action"] == "continue"



    

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8l.pt")

# Full COCO class list (80 classes, indices 0-79)
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

wantedClasses = ["bottle", "cup", "handbag", "cell phone"]

prev_frame_time = 0
new_frame_time = 0
popup_cooldown = 0
last_detected = None

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)

    detected_this_frame = None


    for r in results:
        boxes = r.boxes

    for box in boxes:
        # Class Name
        cls = int(box.cls[0])
        currentClass = classNames[cls]

        # Only process if class is in wantedClasses
        if currentClass in wantedClasses:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            if detected_this_frame is None:
                detected_this_frame = currentClass

    if detected_this_frame is None:
        last_detected = None

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)

    if detected_this_frame and time.time() > popup_cooldown:
        keep_going = pop_up_window(detected_this_frame)
        if not keep_going:
            break
        else:
            last_detected = detected_this_frame
            popup_cooldown = time.time() + 3
            detected_this_frame = None


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

