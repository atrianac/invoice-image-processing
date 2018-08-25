package com.endava.bod.il;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

public class OpenCVTest {

    public static final String RESOURCES_INVOICE_JPEG = "resources/invoice.jpeg";

    public static final String RESOURCES_RESULT_JPEG = "resources/result.jpeg";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
       Mat src = loadImage(RESOURCES_INVOICE_JPEG);
       Mat srcHistogram = equalizeHistogram(src, createDestinationFromSource(src));
       Mat adaptiveThreshold = adaptiveThreshold(srcHistogram, createDestinationFromSource(srcHistogram));
       Mat denoiseImage = deleteNoise(adaptiveThreshold, createDestinationFromSource(adaptiveThreshold));

       writeImage(denoiseImage, RESOURCES_RESULT_JPEG);
    }

    public static Mat deleteNoise(Mat src, Mat dst) {
        Photo.fastNlMeansDenoising(src, dst,50,7,21);
        return dst;
    }

    public static Mat adaptiveThreshold(Mat src, Mat dst) {
        Imgproc.adaptiveThreshold(src, dst, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 40);
        return dst;
    }

    public static Mat writeImage(Mat src, String finalPath) {
        Imgcodecs.imwrite(finalPath, src);
        return src;
    }

    public static Mat equalizeHistogram(Mat src, Mat dst) {
        Imgproc.equalizeHist(src, dst);
        return dst;
    }

    public static Mat createDestinationFromSource(Mat src) {
        return new Mat(src.rows(), src.cols(), src.type());
    }

    public static Mat loadImage(String src) {
        return Imgcodecs.imread(src, 0);
    }

}
