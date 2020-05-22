package com.example.app_cv;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class gender_detect extends face_detect implements CameraBridgeViewBase.CvCameraViewListener2{
    public Net mGenderNet;
    private static final String[] GENDERS = new String[]{"Male", "Female"};
    private static final String TAG = "Test";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.face_detect);
        initDNN();
    }

    private void initDNN() {
        String gender_proto = getPath("deploy_gender.prototxt", this);
        String gender_weights = getPath("gender_net.caffemodel", this);
        Log.i(TAG, "initDNN| Gender Proto : " + gender_proto + ", Gender Weights : " + gender_weights);

        mGenderNet = Dnn.readNetFromCaffe(gender_proto, gender_weights);

        if (mGenderNet.empty()) {
            Log.i(TAG, "Gender Network loading failed");
        } else {
            Log.i(TAG, "Gender Network loading success");
        }
    }

    public String analyzeGender(Mat mRgba, Rect face) {
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

}

