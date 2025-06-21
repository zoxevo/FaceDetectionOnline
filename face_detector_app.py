import cv2
import os
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# بارگذاری مدل‌ها
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(os.path.dirname(__file__), 'models', 'gender_deploy.prototxt'),
    os.path.join(os.path.dirname(__file__), 'models', 'gender_net.caffemodel')
)
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def detect_process(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    out = img_np.copy()

    for (x, y, w, h) in faces:
        if 1.0 <= h / w <= 1.5 and w > 50 and h > 50:
            face = img_np[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            face = cv2.GaussianBlur(face, (45, 45), 30)
            out[y:y+h, x:x+w] = face
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(out, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    return out, len(faces)

# رابط کاربری
st.title("🚀 تشخیص چهره آنلاین")
st.write("عکس را آپلود کنید تا چهره و جنسیت تشخیص داده شود.")

uploaded_file = st.file_uploader("عکس را آپلود کنید", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file).convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    result_img, face_count = detect_process(img)
    
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"{face_count} چهره تشخیص داده شد", use_column_width=True)
    
    buf = BytesIO()
    result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="دانلود تصویر پردازش‌شده",
        data=byte_im,
        file_name="detected_face.png",
        mime="image/png"
    )

if __name__ == "__main__":
    st.run()