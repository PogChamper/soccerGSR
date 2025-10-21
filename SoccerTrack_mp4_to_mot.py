import os
import cv2

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    # Путь к папке с видео
    # video_folder = "/media/viktor/TOSHIBA EXT/DataSets/ScoccerChllenge2025-20251016T205408Z-1-001/ScoccerChllenge2025/Training Dataset"
    video_folder = args.path
    print(f"Data path: {video_folder}")
    # Обработка всех видео в папке
    for video_name in os.listdir(video_folder):
        if video_name.endswith(".mp4"): #and video_name != "117093.mp4":
            video_path = os.path.join(video_folder, video_name)
            
            # Создаем папку для изображений
            output_folder = os.path.join(video_folder, os.path.splitext(video_name)[0])
            os.makedirs(output_folder, exist_ok=True)
            img_folder = os.path.join(output_folder, "img1")
            os.makedirs(img_folder, exist_ok=True)

            # Открываем видео
            cap = cv2.VideoCapture(video_path)
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Извлекаем кадры
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(f"{img_folder}/{frame_count:06d}.jpg", frame)
                frame_count += 1
            cap.release()

            # Создаем файл seqinfo.ini
            seqinfo_path = os.path.join(output_folder, "seqinfo.ini")
            with open(seqinfo_path, "w") as f:
                f.write(
                    f"[Sequence]\n"
                    f"name={os.path.splitext(video_name)[0]}\n"
                    f"imDir=img1\n"
                    f"frameRate={frame_rate}\n"
                    f"seqLength={frame_count}\n"
                    f"imWidth={width}\n"
                    f"imHeight={height}\n"
                    f"imExt=.jpg\n"
                )

if __name__ == "__main__":
    main()