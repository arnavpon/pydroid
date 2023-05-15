## Things to note:
- the "python" folder in this project contains everything relevant to the heart rate estimations
- for an example of how to run/train the model, look at ml.v8.ipynb. Note that it uses the code for augmenting the dataset, which was largely an unsuccessful experiment. You probably want to comment it out and use it without the augmented data.
- VideFaceDetection.py: main entry point to the Python pipeline from the Flutter app, start here for following the HR measurement pipeline
- ml_pipeline_basic.py: puts the pipeline together for HR measurment in a pipeline method called by VideFaceDetection, also a good place to go early in exploring this code
- FaceDetection_MT1.py: Written by Arnav. Contains the wrapper for YoloV5 that we use in the app
- mtl.ipynb: adaptation of the work in "CamSense: A camera-based contact-less heart activity monitoring" by Hasan et al. This is the baseline deep learning model I use in the thesis. Look here at how I use their model and adapt it so that I can use the same cross-validation strategy I employ for my gradient boosting regression model. Note that this is different than the personalized baslien models implemented in deep-learning-files
- deep-learning-files: taken straight from Hasan's work, these train and validate the full CamSense architecture on a single subject; the reason there are so many of them is that each one has to be hardcoded with specific lines chaneged for reading the video to align it with ground truth. LOOK AT THE TRACK.TXT FILE IN THE VALIDATION DATA FOR EACH SUBJECT AS THAT TELLS YOU WHICH LINES TO SWITCH TO MAKE THIS WORK FOR EACH SUBJECT. Note that this isn't a problem in mtl-ipynb, bc after I went through all this hardcoding I pickled the aligned data for each subject for use in mtl.ipynb
- net_work_def.py: taken from Hasan et al's implementation in their repo, https://github.com/mxahan/project_rppg. These are their implementations of various iterations of their architecture. *They don't really make it clear at all which of the models to use without reading very carefully, but the relevant ones are MtlNetwork_body, MtlNetwork_body2, and MtlNetwork_head. The difference between MtlNetwork_body and MtlNetwork_body2 is that MtlNetwork_body is their original implementation of their MTL Body which is only supposed to be connected to an MTL head, whereas MtlNetwork_body2 is my alteration of MtlNetwork_body so that the MTL Body can just output rPPG on its own*; these terms will be more clear if you read the paper
- ml-v8-versions: different versions of the ml.v8 notebook for the augmented data experiment, these can be disregarded
- nn_weights: saved weights from the deep learning experiments! the sub-folders marked "generalized-subX" are the generalized models trained on all subjects except subject X to test how the architecture does in predicting unseen subjects, and the folders just called "subX" are the personalized models for subject X
- replicate_pickle_data: the pickled, aligned data so that the headache of hardcoding starting points for the alignment is removed
- plot_subject_model2.ipynb: for making plots of the results of the deep learning models
- pickle_video_data.py: script for the pickling of the aligned videos and ground truth
- AnalyzeStream.py/FaceDetectionBasic.py/GetHeartrate.py: all deprecated, disregard
- pipeline_v1 is a submodule of the python module containing a lot of the modules used here. here's an overview of what some of the files are:
    - channel_data3, channel_data3_bright, channel_data3_dim, channel_data3_noise: these are the raw, spatially averaged RGB files taken from the subject videos corresponding to the original, brightened, dimmed, and noise-added versions of all the videls; given that the data augmentation didn't really yield better results, channel_data3 (the original spatially averaged RGB) is most relevant to you
    - validation_data/IEEE_data: contains relevant ground truth BVP (amongst other stuff) for each subject. truth.py aligns the ground truth BVP with the RGB data extracted from the videos
    - chrominance.py: implements the chrominance algorithm taken from "Robust pulse rate from chrominance-based rPPG" by De Haan and Jeanne. Didn't work for me in getting HR directly from it, but used it as a feature in the ML model
    - forehead.py: for getting a bounding box for the forehead, once a bounding box for a face has been found
    - ml.v1-7.ipynb: these are all "development" notebooks for the ML and none of them are finished products; you can disregard
    - mtcnn.pb: the YoloV5 model in use
    - optimizing.py: contains the classes I wrote for implementing subjectwise-cross-validation and Bayesian optimization
    - peaks.py: peak detection implementation
    - pipeline_walkthrough.ipynb/pipeline_walkthrough.v1.py/process_ieee.py/rgb_method_comparison.ipynb/rgb_ml_eda.v1.ipynb: all pertaining to earlier versions of the pipeline that are now deprecated
    - roi_feats.py: I was working on a theory that I could potentially do better than just spatially averaging the color channels for the ROI, and this module implemented some feature extraction methods for doing that; but, I ran out of time and I never tested any of this: could be a good avenue for future work
    - ieee_video_start_frames.py: hardcoded start frames of for lining up subject video with ground truth BVP
    - losses.py: implements factory class that yields the loss funcitons for the gradient boosting regressor
    - mb.py: implements the wrapper for the gradient boosting regressor and all the bells and whistles around it
    - signal_pross.py: general signal processing module used throughout the project: contains things like bandpass filtering, convenience methods for things like detrending, calculating IBIs, calculating HR, etc.
    - tracking.py: implements the face tracking and RGB extraction process for producing our RGB features
    - truth.py: factory class that produces subject data aligned with its ground truth BVP
    - wavelet.py: implemementation for wavelet filtering
    - soft_dtw.py: implementation of the Soft-DTW loss function; taken from this GitHub: https://github.com/lyprince/sdtw_pytorch



