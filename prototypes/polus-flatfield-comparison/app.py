import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
from PIL import Image
import tifffile
import os
import glob

dashboard_location = os.path.dirname(__file__)

#Title 
st.title("Polus Flat-Field Comparison App")


#App Menu
selected = option_menu(None, ["App", "About"],
    icons=[ 'app','info-square'],
    default_index=0, orientation="horizontal")


#About Page 
if selected == "About":
    #Set page title
    st.header("About the App")
    text = """
    The image dataset used in this project includes a collection of images provided by Axle's Polus AI team. To correct these images, the Polus AI Plugins for Flat-field Corrections were used, including the estimation and apply plugins. The raw images were processed, resulting in corrected images that were stored in a designated output folder to be used in this app.

    The Polus Flat-Field Comparison App was developed to showcase the effectiveness of the correction process on the images. This app utilizes the Streamlit Image Comparison component from the `streamlit-image-comparison` library. It enables side-by-side comparison and visualization of two images, allowing users to easily observe differences and analyze the impact of the correction.

    By showcasing the correction performed by the Polus AI Plugins with the Streamlit Image Comparison component, this project provides a tool for evaluating and understanding the effects of flat-field correction on the image dataset.
       """
    st.markdown(text)

    #FCC Example Image
    img_path = dashboard_location + '/data/example-FFC.png'
    fcc_example = Image.open(img_path)
    st.image(fcc_example, use_column_width=True, caption="Flat-field correction was used to cancel out effects of image artifacts or noise, resulting in a uniformly-illuminated image and an improvement in the overall quality of the image.")


    expander = st.expander("Expand to see links to Sources/Tools")
    with expander:
        st.markdown("Sources:")
        st.markdown("- [Wikipedia - Flat-field Correction](https://en.wikipedia.org/wiki/Flat-field_correction)")
        st.markdown("- [CALM UCSF - Flat-field Correction](https://calm.ucsf.edu/how-acquire-flat-field-correction-images)")
        st.markdown("Tools:")
        st.markdown("- [Streamlit Image Comparison Component](https://pypi.org/project/streamlit-image-comparison/)")
        st.markdown("- [Polus AI Flat-field Estimation Plugin ](https://github.com/PolusAI/polus-plugins/tree/master/regression/polus-basic-flatfield-correction-plugin)")
        st.markdown("- [Polus AI Apply Flat-field Plugin](https://github.com/PolusAI/polus-plugins/tree/master/transforms/images/polus-apply-flatfield-plugin)")
        st.write('Note - The Polus AI Flat-field Estimation Plugin is currently being updated. The link provided is to the older version of the plugin. Both plugins can be run locally in Docker.')

#App Page 
elif selected == "App":
    #Info text for select input
    info = "The corrected images were previously generated using the Polus AI Flat-Field Correction plugins."

    #Path to the folder containing image files
    folder_path = dashboard_location + "/data/original/"

    #List of file names from the folder
    file_names = [os.path.basename(f) for f in glob.glob(f"{folder_path}/*.ome.tif")]

    with st.sidebar:
        st.subheader("Select an Image to Compare:")
        #Display the input selectbox with the file names
        selected_file = st.selectbox("Select a File", file_names, index=1, help=info)
        st.sidebar.markdown("Notice that the raw image quality is less than ideal. Due to uneven illumination and detection, some regions of the image, typically the center, are brighter, and others are dimmer.")
        st.sidebar.markdown("[Flat-Field Correction](https://calm.ucsf.edu/how-acquire-flat-field-correction-images) was applied to correct the image and cancel out effects of image artifacts or noise, resulting in a uniformly-illuminated image and an improvement in the overall quality of the image.")

    if selected_file:
        st.subheader("Selected Image:  " + selected_file)

        #Set the paths to OME-TIFF files
        img_path1 = dashboard_location + "/data/original/" + selected_file
        img_path2 = dashboard_location + "/data/corrected/" + selected_file

        #Read OME-TIFF files using tifffile
        img1 = tifffile.imread(img_path1)
        img2 = tifffile.imread(img_path2)


        #Display the image comparison using Streamlit Image Comparison Component 
        image_comparison(
            img1=img1,
            img2=img2,
            label1="Raw",
            label2="Corrected"
        )

    else:
        st.write("No files selected.")
