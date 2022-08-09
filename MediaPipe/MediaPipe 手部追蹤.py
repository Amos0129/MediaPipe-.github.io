import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands #手部追蹤模型
#(是否靜態圖片還是動態圖片,偵測幾隻手,模型複雜度0或1(越大越精準),最低偵測嚴謹度((0~1)越大越嚴謹),追蹤嚴謹度((0~1)越大越嚴謹))
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5) #設定點的顏色 粗度 半徑
handConStyle = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=10) #設定線的顏色 粗度 半徑

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR轉RGB
        result = hands.process(imgRGB)
        #print(result.multi_hand_landmarks) #顯示21個點的座標

        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:  #找每一隻手的線
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle) #(畫在哪張圖,手部點畫出來,所有點連接起來,點的樣式,線的樣式)
                for i, lm in enumerate(handLms.landmark): #找每一隻手的點
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2) #每一個點在手的哪個位置
                    if i == 4:
                        cv2.circle(img, (xPos, yPos), 20, (99, 87, 87), cv2.FILLED) #把第四個點放大出來
                    #print(i, xPos, yPos) #每個點的X,Y座標

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('img', img) #每一偵顯示出來

    if cv2.waitKey(1) == ord('q'):
        break