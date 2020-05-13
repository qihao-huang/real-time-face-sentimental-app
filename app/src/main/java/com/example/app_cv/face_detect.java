package com.example.app_cv;

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class face_detect extends AppCompatActivity implements
        CameraBridgeViewBase.CvCameraViewListener2 {

    private CameraBridgeViewBase cameraView;
    private CascadeClassifier classifier;
    private Mat mGray;
    private Mat mRgba;
    private int mAbsoluteFaceSize = 0;

    private static final String TAG = "Test";

    private Net mAgeNet;
    private static final String[] AGES =
            new String[]{"0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+"};

    private Net mGenderNet;
    private static final String[] GENDERS = new String[]{"Male", "Female"};

    private static final String[] EMOTIONS =
            new String[]{"angry", "disgust", "fear", "happy", "sad", "surprise"};
    // 0-1-2-3-4-5-6

    // JNI
    static {
        System.loadLibrary("opencv_java3");
    }

    static {
        System.loadLibrary("tensorflow_inference");
    }

    // sentimental variables initialization
    private String MODEL_PATH = "file:///android_asset/sentimental_model.pb";
    private String INPUT_NAME = "convolution2d_10_input";
    private String OUTPUT_NAME = "dense_6/Softmax";

    private TensorFlowInferenceInterface tf;

    float[] PREDICTIONS = new float[6];

    private static final int HEIGHT = 48;
    private static final int WIDTH = 48;
    private static final int CHANNEL = 1;
    private float[] INPUT_DATA = new float[HEIGHT*WIDTH*CHANNEL];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        initWindowSettings();
        setContentView(R.layout.face_detect);
        cameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
//        cameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        cameraView.setCvCameraViewListener(this);
        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT); // use front camera
        cameraView.enableView();
        // -------------------------
        initClassifier();
        initDNN();
        initTF();
        // -------------------------
    }

    private void initWindowSettings() {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED);
    }

    // ------------------------------------------------

    private void initTF(){
        tf = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);
    }

    private String analyzeSentiment(final Mat mGray, final Rect face) {
        try{
            Mat capturedFace = new Mat(mGray, face);
            Imgproc.resize(capturedFace, capturedFace, new Size(WIDTH, HEIGHT));

            INPUT_DATA = mat_to_array(capturedFace);
            //Pass input into the tensorflow
            tf.feed(INPUT_NAME, INPUT_DATA, 1, WIDTH, HEIGHT, CHANNEL);
            tf.run(new String[]{OUTPUT_NAME}); //compute predictions
            tf.fetch(OUTPUT_NAME, PREDICTIONS); //copy the output into the PREDICTIONS array

            Log.i(TAG, Arrays.toString(PREDICTIONS));

        } catch (Exception e) {
            Log.e(TAG, "Error processing sentiment", e);
        }

        return null;

//        Object[] results = argmax(PREDICTIONS); //Obtained highest prediction

//        int class_index = (Integer) results[0];
//        final String pred_emotion = EMOTIONS[class_index];

//        return pred_emotion;
    }

    private float[] mat_to_array(Mat imgMat){
        Imgproc.cvtColor(imgMat, imgMat, Imgproc.COLOR_GRAY2RGBA, 4);
        Bitmap bmp = Bitmap.createBitmap(imgMat.cols(), imgMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imgMat, bmp);

        float[] output = new float[HEIGHT*WIDTH*CHANNEL];
        int[] intValues = new int[bmp.getHeight() * bmp.getWidth()];
        bmp.getPixels(intValues, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            output[i] = val & 0xFF;
        }

        return output;
    }

//    public static Object[] argmax(float[] array) {
//        int best = -1;
//        float best_confidence = 0.0f;
//        for (int i = 0; i < array.length; i++) {
//            float value = array[i];
//            if (value > best_confidence) {
//                best_confidence = value;
//                best = i;
//            }
//        }
//        return new Object[]{best, best_confidence};
//    }

    // ------------------------------------------------

    private void initDNN() {
        // ------------------------------------------------
        String age_proto = getPath("deploy_age.prototxt", this);
        String age_weights = getPath("age_net.caffemodel", this);
        Log.i(TAG, "initDNN| Age Proto : " + age_proto + ", Age Weights : " + age_weights);

        mAgeNet = Dnn.readNetFromCaffe(age_proto, age_weights);

        // ------------------------------------------------

        String gender_proto = getPath("deploy_gender.prototxt", this);
        String gender_weights = getPath("gender_net.caffemodel", this);
        Log.i(TAG, "initDNN| Gender Proto : " + gender_proto + ", Gender Weights : " + gender_weights);

        mGenderNet = Dnn.readNetFromCaffe(gender_proto, gender_weights);

        // ------------------------------------------------

        if (mAgeNet.empty()) {
            Log.i(TAG, "Age Network loading failed");
        } else {
            Log.i(TAG, "Age Network loading success");
        }

        if (mGenderNet.empty()) {
            Log.i(TAG, "Gender Network loading failed");
        } else {
            Log.i(TAG, "Gender Network loading success");
        }
    }

    // ------------------------------------------------

    private void initClassifier() {
        try {
            // Load LBP front face detector
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_APPEND);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            classifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ------------------------------------------------

    private String analyzeAge(Mat mRgba, Rect face) {
        try {
            Mat capturedFace = new Mat(mRgba, face);
            //Resizing pictures to resolution of Caffe model
            Imgproc.resize(capturedFace, capturedFace, new Size(227, 227));
            //Converting RGBA to BGR
            Imgproc.cvtColor(capturedFace, capturedFace, Imgproc.COLOR_RGBA2BGR);

            //Forwarding picture through Dnn
            Mat inputBlob = Dnn.blobFromImage(capturedFace, 1.0f,
                    new Size(227, 227),
                    new Scalar(78.4263377603, 87.7689143744, 114.895847746),
                    false, false);

            mAgeNet.setInput(inputBlob, "data");
            Mat probs = mAgeNet.forward("prob").reshape(1, 1);
            Core.MinMaxLocResult mm = Core.minMaxLoc(probs); //Getting largest softmax output

            double result = mm.maxLoc.x; //Result of age recognition prediction
            Log.i(TAG, "Age result is: " + result);
            return AGES[(int) result];
        } catch (Exception e) {
            Log.e(TAG, "Error processing age", e);
        }
        return null;
    }

    private String analyzeGender(Mat mRgba, Rect face) {
        try {
            Mat capturedFace = new Mat(mRgba, face);
            //Resizing pictures to resolution of Caffe model
            Imgproc.resize(capturedFace, capturedFace, new Size(227, 227));
            //Converting RGBA to BGR
            Imgproc.cvtColor(capturedFace, capturedFace, Imgproc.COLOR_RGBA2BGR);

            //Forwarding picture through Dnn
            Mat inputBlob = Dnn.blobFromImage(capturedFace, 1.0f,
                    new Size(227, 227),
                    new Scalar(78.4263377603, 87.7689143744, 114.895847746),
                    false, false);

            mGenderNet.setInput(inputBlob, "data");
            Mat probs = mGenderNet.forward("prob").reshape(1, 1);
            Core.MinMaxLocResult mm = Core.minMaxLoc(probs); //Getting largest softmax output

            //Result of gender recognition prediction. 0 = MALE, 1 = FEMALE
            double result = mm.maxLoc.x;
            Log.i(TAG, "Gender result is: " + result);
            return GENDERS[(int) result];
        } catch (Exception e) {
            Log.e(TAG, "Error processing gender", e);
        }
        return null;
    }

    // Upload file to storage and return a path.
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            Log.i(TAG, outFile.getAbsolutePath());
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        // -------Rotate the preview the streaming-------
        Mat rotateMat = Imgproc.getRotationMatrix2D(
                new Point(mGray.rows() / 2, mGray.cols() / 2), 90, 1);
        Imgproc.warpAffine(mRgba, mRgba, rotateMat, mRgba.size());
        Imgproc.warpAffine(mGray, mGray, rotateMat, mGray.size());

        Core.flip(mRgba, mRgba, 1);
        Core.flip(mGray, mGray, 1);
        // -------Rotate the preview the streaming-------

        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        // mAbsoluteFaceSize is nearly 20% of height. (rows)

        MatOfRect faces = new MatOfRect();

        // face detector
        if (classifier != null) {
            classifier.detectMultiScale(mGray, faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else {
            Log.e(TAG, "Detection is not selected!");
        }

        // plot rectangle for each detected face
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255);
        for (Rect faceRect : facesArray) {
            String predict_age = "";
            String predict_gender = "";
            String predict_emotion = "";
            try{
                predict_age = analyzeAge(mRgba, faceRect);
                predict_gender = analyzeGender(mRgba, faceRect);
                predict_emotion = analyzeSentiment(mGray, faceRect);
            } catch (Exception e) {
                Log.e(TAG, "Error", e);
            }

            Imgproc.rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3);
//            // TODO: change emotion position
//            Imgproc.putText(mRgba, predict_emotion, new Point(mGray.rows() / 2, mGray.cols() / 2),
//                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(0, 255, 255), 4);
            Imgproc.putText(mRgba, predict_age, faceRect.tl(),
                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 0), 4);
            Imgproc.putText(mRgba, predict_gender, faceRect.br(),
                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 0), 4);
        }

        return mRgba;
    }

    // -----------------------------------
    @Override
    public void onPause() {
        super.onPause();
        if (cameraView != null)
            cameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraView.disableView();
    }
    // -----------------------------------
}
