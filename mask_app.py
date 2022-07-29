from re import M
from detectron2.data.datasets import register_coco_instances
import cv2
import numpy as np
import streamlit as st
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#ВАЖНО!!!
#Загружаем датасет, размеченный с помощью Roboflow

# !pip install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YSXE7WwzEvf1S2uNYS6O")
# project = rf.workspace("dselbrus").project("mask_no_mask-fw9uj")
# dataset = project.version(10).download("coco")

#Загружаем натренированную модель из GoogleDrive
#https://drive.google.com/file/d/1RE3No0ww8ptTDuQ6AIUMfqC-tQHcJ9xd/view?usp=sharing


@st.cache(persist=True)
def initialization():
    #создаём модель с сохранёнными параметрами
    register_coco_instances("mask_train", {}, 
                         "/home/anna/ds_bootcamp/ds_offline/model_mask_no_mask/Mask_no_mask-10/train/_annotations.coco.json", 
                         "/home/anna/ds_bootcamp/ds_offline/model_mask_no_mask/Mask_no_mask-10/train")
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/anna/ds_bootcamp/ds_offline/model_mask_no_mask/model_final.pth")  # путь к обученной модели
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    # устанавливаем порог обнаружения
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85

    # создаем объект для построения предсказаний
    predictor = DefaultPredictor(cfg)

    return cfg, predictor

class Metadata:
    def get(self, _):
        return ['mask-detection', 'mask_weared_incorrect', 'with_mask', 'without_mask']


@st.cache
def inference(predictor, img):
    return predictor(img)


@st.cache
def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], Metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()

    return processed_img

def main():
    # Initialization
    cfg, predictor = initialization()

    # Streamlit initialization
    st.title('Распознавание лица в маске и без маски с Detectron2 Fast R-CNN X101-FPN')
    st.write('Наша задача - детекция объектов. Нам необходимо выделить несколько объектов на изображении через нахождение координат их ограничивающих рамок и классифировать этих ограничивающих рамок из множества заранее известных классов.')
    st.image('images/output_example2.png')

    # Retrieve image
    uploaded_img = st.file_uploader("ЗАГРУЗИ КАРТИНКУ СЮДА", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # Detection code
        outputs = inference(predictor, img)
        out_image = output_image(cfg, img, outputs)
        st.image(out_image, caption='Надо подождать...', use_column_width=True)        


if __name__ == '__main__':
    main()