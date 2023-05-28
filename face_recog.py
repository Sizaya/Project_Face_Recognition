import face_recognition
import os
import cv2
import training

KNOWN_FACES_DIR = 'known_face'
video = cv2.VideoCapture('/home/nataliia/Documents/Py/DZ_Nataliia_Sizinevska/Project/video/presidents480.mp4')
TOLERANCE = 0.6         # Допустима межа відхилення
FRAME_THICKNESS = 3     # Товщина рамки
FONT_THICKNESS = 1      # Товщина контуру для тексту
MODEL = training        # /cnn


print('Завантаження відомих облич...')
known_faces = []
known_names = []

# Ми упорядковуємо відомі обличчя як підпапки KNOWN_FACES_DIR
# Назва кожної вкладеної папки стає нашою міткою (name)
for name in os.listdir(KNOWN_FACES_DIR):
    # Завантажуємо кожен файл облич відомої особи
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # Завантажити зображення
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # Отримати 128-вимірне кодування обличчя
        # Завжди повертає список знайдених облич, для цього ми беремо
        # лише перше обличчя (припускаємо, що одне обличчя на зображення,
        # оскільки ви не можете бути двічі на одному зображенні)
        encoding = face_recognition.face_encodings(image)[0]
        # Додати кодування та ім'я
        known_faces.append(encoding)
        known_names.append(name)

while True:
    ret, image = video.read()
    # Цього разу ми спочатку захоплюємо розташування граней
    locations = face_recognition.face_locations(image, model=MODEL)
    # Оскільки ми знаємо розташування, ми можемо передати їх face_encodings як другий аргумент
    # Без цього він знову шукатиме обличчя, уповільнюючи весь процес
    encodings = face_recognition.face_encodings(image, locations)
    # Але цього разу ми припускаємо, що на зображенні може бути більше облич - ми можемо знайти обличчя різних людей
    print(f'Знайдено {len(encodings)} облич(чя)')
    for face_encoding, face_location in zip(encodings, locations):
        # Ми використовуємо compare_faces (але також можемо використовувати face_distance)
        # Повертає масив значень True/False у порядку переданих
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        # Оскільки порядок зберігається, ми перевіряємо, чи знайдено будь-яке обличчя, потім беремо індекс
        # потім мітку (ім’я) першого відповідного відомого обличчя з відповідністю допуску
        match = None
        if True in results:  # Якщо хоча б одна вірна, отримайте назву першої зі знайдених міток
            match = known_names[results.index(True)]
            print(f'Збіг знайдено: {match}')
            # Кожне розташування містить позиції в порядку: зверху, справа, знизу, зліва
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 0, 0]
            # Зафарбувати кадр
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            # Тепер нам потрібна зафарбована ділянка для імені
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            # Фарбування кадру
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            # Визначення імені
            cv2.putText(image, match, (face_location[3] + 5, face_location[2] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
    # Показати зображення
    cv2.imshow('', image)
    if cv2.waitKey(30) & 0xFF == 27:
        break

