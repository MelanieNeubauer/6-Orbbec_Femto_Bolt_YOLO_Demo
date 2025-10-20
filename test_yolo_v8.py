from ultralytics import YOLO
import numpy as np
import cv2
import sqlite3
import glob
import os


def get_result_yolo_v8_image(orig_image, model):
    """ Get a result for one single image with your model """
    im_size = orig_image.shape
    results = model.predict(orig_image)
    masks = results[0].masks

    for mask in masks:
        mask_data = mask.cpu().data[0].numpy()
        mask_data = cv2.resize(mask_data, (im_size[1], im_size[0]))

        mask_data[mask_data > 0] = 255
        mask_data = mask_data.astype('uint8')

        red_img = np.zeros(orig_image.shape, orig_image.dtype)
        red_img[:, :] = (0, 0, 255)
        red_mask = cv2.bitwise_and(red_img, red_img, mask=mask_data)
        cv2.addWeighted(red_mask, 1, orig_image, 1, 0, orig_image)

        polygon = mask.xy[0].astype('int32').reshape((-1, 1, 2))
        image = cv2.polylines(orig_image, pts=[polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    return image


def create_video_yolo_v8(path_video, path_output_video, model):
    """ Creates one single output video from and input video with predicted masks from a model."""
    cap = cv2.VideoCapture(path_video)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        try:
            pred = get_result_yolo_v8_image(frame, model)
        except:
            pred = frame

        # Write the processed frame to the output video
        out.write(pred)

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def create_database_from_video(path_database, path_video, model, object_class):
    """ Creates one Database for the input video with the given model. It saves the objects with one single Class."""

    # Connection to Video Capture
    cap = cv2.VideoCapture(path_video)
    current_frame = 0

    # Connection to SQLite-Database
    db_conn = sqlite3.connect(path_database)
    db_cursor = db_conn.cursor()

    # Create needed tables in database
    db_cursor.execute('''
        CREATE TABLE labels (
            frame INTEGER,
            class TEXT,
            code TEXT
        )
    ''')

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        try:
            pred = model.predict(frame)
            pred = pred[0].masks
        except:
            pred = None

        if pred is None:
            current_frame += 1
            continue

        for mask in pred:
            poly = mask.xy[0]      # ndarray(dx, dy)
            # Convert the ndarray to a list of tuples
            poly_list = [(point[0], point[1]) for point in poly]
            db_cursor.execute(f'''INSERT INTO labels VALUES ({current_frame}, '{object_class}', '{str(poly_list)}')''')

        current_frame += 1

    # Release everything when done
    db_conn.commit()
    db_cursor.close()
    db_conn.close()

    cap.release()
    cv2.destroyAllWindows()

    print('Done!')


if __name__ == '__main__':
    #model_yolo = YOLO('../Data/Pretrained_Models/runs-copper-detection/1-4080-PC2/weights/best.pt')
    model_yolo = YOLO('../Data/Pretrained_Models/F7_a1-train_32-860_auto_w16/weights/best.pt')

    #image = cv2.imread("image.jpg")
    #get_result_yolo_v8(image)

    video_folder_path  = ('/media/melanie/cps4TB/Aufnahme 3/Aufnahmen GoPro REDUCED')
    video_path_list = glob.glob(os.path.join(video_folder_path, '*.MP4'))
    video_path_list = sorted(video_path_list)
    database_dir = ('/media/melanie/cps4TB/Aufnahme 3/Aufnahmen GoPro REDUCED Databases/')


    for video_path in video_path_list:
        database_path = database_dir + os.path.splitext(os.path.basename(video_path))[0] + '.db'

        if os.path.exists(database_path):
            continue

        create_database_from_video(database_path, video_path,
                                   model_yolo, 'Object')

    #video_path = '../Data/Videos/2024_02_29 Aufnahme 2/a1/a1_train_5.mp4'  # Provide the path to your video file
    #output_path = "../Results/video1.mp4"
    #create_video_yolo_v8(video_path, output_path, model_yolo)

