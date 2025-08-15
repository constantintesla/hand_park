import argparse
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from ultralytics import YOLO

def analyze_video(video_path, output_csv="hand_data.csv", show_video=True):
    # Инициализация моделей
    yolo_model = YOLO('yolov8n.pt')  # YOLO для обнаружения людей
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка открытия видеофайла")
        return
    
    # Подготовка для сохранения данных
    hand_data = []
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Обнаружение людей с помощью YOLO
        yolo_results = yolo_model(frame, classes=[0])  # класс 0 - человек
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            # Берем первого обнаруженного человека (можно модифицировать для нескольких)
            x1, y1, x2, y2 = map(int, boxes[0])
            person_roi = frame[y1:y2, x1:x2]
            
            # Обнаружение рук в области человека
            results = hands.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Получаем координаты ключевых точек кисти
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x = x1 + landmark.x * (x2 - x1)
                        y = y1 + landmark.y * (y2 - y1)
                        z = landmark.z
                        landmarks.extend([x, y, z])
                    
                    # Сохраняем данные для текущего кадра
                    hand_data.append([frame_count, timestamp] + landmarks)
                    
                    # Визуализация
                    if show_video:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if show_video:
            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Закрытие ресурсов
    cap.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # Сохранение данных в CSV
    if hand_data:
        columns = ['frame', 'timestamp']
        for i in range(21):  # 21 ключевая точка в MediaPipe Hands
            columns.extend([f'hand_{i}_x', f'hand_{i}_y', f'hand_{i}_z'])
        
        df = pd.DataFrame(hand_data, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Данные сохранены в {output_csv}")
    else:
        print("Не обнаружено рук на видео")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Анализ движения кисти человека на видео с сохранением в CSV')
    parser.add_argument('--input', required=True, help='Путь к входному видеофайлу')
    parser.add_argument('--output', default="hand_data.csv", help='Путь для сохранения CSV (по умолчанию: hand_data.csv)')
    parser.add_argument('--no-display', action='store_true', help='Не показывать видео в реальном времени')
    
    args = parser.parse_args()
    
    analyze_video(
        video_path=args.input,
        output_csv=args.output,
        show_video=not args.no_display
    )