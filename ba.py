import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_option_menu import option_menu

image = Image.open('ba_logo1.png')
st.image(image,use_column_width=True)
step = Image.open('BA_step.png')

EXAMPLE_NO = 1

def streamlit_menu(example=1):
    selected = option_menu(
        menu_title=None,  # required
        options=["ホーム", "使い方"],  # required
        icons=['house', 'bi-file-earmark-text'],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
    )
    return selected

selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "ホーム":
    # メイン画面
    # ラジオボタン
    st.header("向き")
    muki=st.radio("向きについては「使い方」をご覧ください",('右向き', '左向き'))
    print(muki);
    # ラジオボタン
    #st.radio("反射",('反射あり', '反射なし'),key = "radio2")

    def scale_to_height(img, height):
        """高さが指定した値になるように、アスペクト比を固定して、リサイズする。
        """
        h, w = img.shape[:2]
        width = round(w * (height / h))
        dst = cv2.resize(img, dsize=(width, height))

        return dst

    col1, col2= st.columns(2)
    image_loc = st.empty()
    col1.header("表面")
    col2.header("側面")
    with col1:
        uploaded_image1=st.file_uploader("表面画像アップ", type = (["jpg", "jpeg", "png"]), key = "up1")
        if uploaded_image1 is not None:
            image1=Image.open(uploaded_image1,)
            img_array1 = np.array(image1)
            img_array1=cv2.cvtColor(img_array1, cv2.COLOR_RGB2RGBA)
            dst1 = scale_to_height(img_array1, 300)
            st.image(dst1,caption = '表面',use_column_width = 'auto')
            
    with col2:
        uploaded_image2=st.file_uploader("側面画像アップ", type = (["jpg", "png", "jpeg"]),  key = "ups2")
        if uploaded_image2 is not None:
            image2=Image.open(uploaded_image2,)
            img_array2 = np.array(image2)
            img_array2=cv2.cvtColor(img_array2, cv2.COLOR_RGB2RGBA)
            dst2 = scale_to_height(img_array2, 300)
            st.image(dst2,caption = '側面', use_column_width = 'auto')
            
    if st.button("実行！",key = "b1"):
        if(muki=='左向き'):
            #表メン
            height, width, channels = img_array1.shape[:3]
            ti=0.1;
            ti2=0.5;
            yu=0.025;
            yu2=0.06;
            source_points1 = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
            target_points1 = np.array([[ti*width, yu*1.3*height], [ti*width, (1-yu*0.8)*height], [width, height], [width, 0]], dtype=np.float32)
            mat = cv2.getPerspectiveTransform(source_points1, target_points1)
            perspective_image1 = cv2.warpPerspective(img_array1, mat, (width, height))
            
            #側面
            height2, width2, channels2 = img_array2.shape[:3]
            width22=(1-ti2)*width2
            #暗く
            #ブランク画像
            blank = np.zeros_like(img_array2)
            blank[:,:,3]=255;
            print(blank.shape)
            print(img_array2.shape)

            k=0.6;
            k_img = cv2.addWeighted(blank,k,img_array2,1-k,0)
            
            source_points2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]], dtype=np.float32)
            target_points2 = np.array([[0, 0],  [0, height], [(1-ti2)*width2, (1-yu2)*height], [(1-ti2)*width2, yu2*height]], dtype=np.float32)
            mat2 = cv2.getPerspectiveTransform(source_points2, target_points2)
            # perspective_image2 = cv2.warpPerspective(img_array2, mat2, (width, height))
            perspective_image2 = cv2.warpPerspective(k_img, mat2, (width2, height))
    
            #合成
            im_h = cv2.hconcat([perspective_image1,perspective_image2])
            st.image(im_h,caption = '右クリックから保存してください',use_column_width = None)
        if(muki=='右向き'):
            #表メン
            height, width, channels = img_array1.shape[:3]
            ti=0.1;
            ti2=0.5;
            yu=0.025;
            yu2=0.06;
            print(width);
            source_points1 = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
            target_points1 = np.array([[0, 0],  [0, height], [(1-ti)*width, (1-yu*0.8)*height], [(1-ti)*width, yu*1.3*height]], dtype=np.float32)
            mat = cv2.getPerspectiveTransform(source_points1, target_points1)
            perspective_image1 = cv2.warpPerspective(img_array1, mat, (width, height))
            
            #側面
            height2, width2, channels2 = img_array2.shape[:3]
            #暗く
            #ブランク画像
            blank = np.zeros_like(img_array2)
            blank[:,:,3]=255;
            print(blank.shape)
            print(img_array2.shape)

            k=0.6;
            k_img = cv2.addWeighted(blank,k,img_array2,1-k,0)
            
            source_points2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]], dtype=np.float32)
            target_points2 = np.array([[ti2*width2, yu2*height], [ti2*width2, (1-yu2)*height], [width2, height], [width2, 0]], dtype=np.float32)
            mat2 = cv2.getPerspectiveTransform(source_points2, target_points2)
            perspective_image2 = cv2.warpPerspective(k_img, mat2, (width2, height))

            #合成
            im_h = cv2.hconcat([perspective_image2,perspective_image1])
            st.image(im_h,caption = '右クリックから保存してください',use_column_width = None)
    

if selected == "使い方":
    st.video('howtouseba.mp4', format="video/mp4", start_time=0)
    st.image(step,caption = '使い方',use_column_width=True)
    muki_img = Image.open('muki_img.png')
    st.image(muki_img,caption = '向きについて',use_column_width=True)





    


