import cv2
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
from datetime import datetime


cascade_path = ".env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
img_path = "./004.jpg"
out_path = "./"


def faces_detection(src):
    '''Effectue une détection des visages et retourne le résultat de la détection et les cadres de reconnaissance'''

    cascade = cv2.CascadeClassifier(cascade_path)
    rects = cascade.detectMultiScale(src)

    for i, [x, y, w, h] in enumerate(rects):
        text_id = f"Personne {i}"
        src = cv2.rectangle(src, (x, y), (x+w, y+h), color=-1)
        src = cv2.rectangle(src, (x, y - 20), (x+w, y), color=-1, thickness=-1)
        src = cv2.putText(src, text_id, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    
    return src, rects

if __name__ == "__main__":

    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.saved = False
    
    st.title("Face Detection App")
    st.text("Build with Streamlit and OpenCV")

    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

        enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale"])

        if enhance_type == "Gray-Scale":
            image_enhanced = cv2.cvtColor(np.array(our_image), cv2.COLOR_BGR2GRAY)
        else:
            image_enhanced = np.array(our_image)

        tasks = ["Faces"]

        feature_choice = st.sidebar.selectbox("Find Features", tasks)

        if st.button("Process") or st.session_state.processed:
            st.session_state.processed = True

            if "saved" not in st.session_state:
                st.session_state.saved = False

            if feature_choice == "Faces":
                image_processed, rects = faces_detection(image_enhanced)
                st.image(image_processed)
                st.success(f"Found {len(rects)} faces !")
            else:
                st.text("Select a feature to detect.")

            now = datetime.now()

            date = now.date().strftime("%d/%m/%Y")
            time = now.time().strftime("%H:%M")

            n_rect = len(rects)

            data = {
                "Date":[date]*n_rect,
                "Time":[time]*n_rect,
                "Personne":[f"Persone {i}" for i in range(1, n_rect + 1)]
            }

            df = pd.DataFrame(data)

            st.table(df)

        if st.session_state.saved:
            st.success("Data saved !")
            if st.button("New"):                 
                st.session_state.process = False                
                # st.session_state.saved = False

        elif st.session_state.processed and not st.session_state.saved:
            # button_save = st.button("Save")
            if st.button("Save"):
                st.session_state.saved = True
                # button_save = st.empty()
                csv_filename = "log_detection.csv"
                try:
                    data = pd.read_csv(out_path + csv_filename)
                    df = data.append(df)
                    df.to_csv(out_path + csv_filename, index=False)
                except FileNotFoundError:
                    df.to_csv(out_path + csv_filename, index=False)
            
            

    # src = cv2.imread(img_path)
    # src, rects = faces_detection(src)
    # cv2.imshow("Detection de visages", src)
    # cv2.waitKey()
    # cv2.destroyAllWindows()