package com.example.app_cv;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.util.Log;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Arrays;

public class sentiment_detect extends face_detect implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String[] EMOTIONS =
            new String[]{"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"};
    static {
        System.loadLibrary("tensorflow_inference");
    }

    // sentimental variables initialization
    private String MODEL_PATH = "file:///android_asset/sentimental_model.pb";
    private String INPUT_NAME = "conv2d_29_input";
    private String OUTPUT_NAME = "activation_48/Softmax";

    private TensorFlowInferenceInterface tf;

    public void initTF(){
        tf = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);
    }

    float[] PREDICTIONS = new float[7];
    private static final int HEIGHT = 48;
    private static final int WIDTH = 48;
    private static final int CHANNEL = 1;

    public String analyzeSentiment(final Mat mGray, final Rect face) {
        initTF();
        try{
            Mat capturedFace = new Mat(mGray, face);

            Imgproc.cvtColor(capturedFace, capturedFace, Imgproc.COLOR_GRAY2RGBA, 4);
            Bitmap bmp = Bitmap.createBitmap(capturedFace.cols(), capturedFace.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(capturedFace, bmp);
            Bitmap grayImage = toGrayscale(bmp);
            Bitmap resizedImage = getResizedBitmap(grayImage,WIDTH,HEIGHT);

            int[] pixelArray = new int[resizedImage.getWidth()*resizedImage.getHeight()];
            resizedImage.getPixels(pixelArray, 0, resizedImage.getWidth(), 0, 0,
                    resizedImage.getWidth(), resizedImage.getHeight());
            float[] normalized_pixels = new float[pixelArray.length];

            for (int i=0; i < pixelArray.length; i++) {
                // 0 for white and 255 for black
                int b = pixelArray[i] & 0xff;

                normalized_pixels[i] = (float)(b);
            }

//            Log.d("pixel_values", String.valueOf(normalized_pixels[0]));

            //Pass input into the TensorFlow
            tf.feed(INPUT_NAME, normalized_pixels, 1, WIDTH, HEIGHT, CHANNEL);
            tf.run(new String[]{OUTPUT_NAME}); //compute predictions
            tf.fetch(OUTPUT_NAME, PREDICTIONS); //copy the output into the PREDICTIONS array

            Log.i(TAG, Arrays.toString(PREDICTIONS));

        } catch (Exception e) {
            Log.e(TAG, "Error processing sentiment", e);
        }

        Object[] results = argmax(PREDICTIONS); //Obtained highest prediction
        int class_index = (Integer) results[0];
        final String pred_emotion = EMOTIONS[class_index];

        Log.i(TAG, "pred_emotion: " + pred_emotion);

        return pred_emotion;
    }

    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);

        return bmpGrayscale;
    }

    public Bitmap getResizedBitmap(Bitmap image, int bitmapWidth, int bitmapHeight) {
        return Bitmap.createScaledBitmap(image, bitmapWidth, bitmapHeight, true);
    }

    public static Object[] argmax(float[] array) {
        int best = -1;
        float best_confidence = 0.0f;
        for (int i = 0; i < array.length; i++) {
            float value = array[i];
            if (value > best_confidence) {
                best_confidence = value;
                best = i;
            }
        }
        return new Object[]{best, best_confidence};
    }

}
