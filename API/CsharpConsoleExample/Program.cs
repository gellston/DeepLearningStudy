using System;
using OpenCvSharp;



namespace CsharpConsoleExample
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Mat original = new Mat(@"source.jpg", OpenCvSharp.ImreadModes.Grayscale);
                Mat resultOverlay = new Mat(original.Width, original.Height, MatType.CV_8UC3);
                Mat resizedInputImage = new Mat(512, 512, MatType.CV_8UC1);
                Mat resized_output_probability = new Mat(512, 512, MatType.CV_32FC1);
                Mat resized_output_threshold = new Mat(512, 512, MatType.CV_32FC1);
                Mat resized8Bit_output_threshold = new Mat(512, 512, MatType.CV_8UC1);




                Cv2.Resize(original, resizedInputImage, new Size(512, 512));
                Cv2.CvtColor(original, resultOverlay, ColorConversionCodes.GRAY2BGR);






                HV.V1.DEEP.Segmentation segmentation = new HV.V1.DEEP.Segmentation();
                segmentation.Import("C://Github//DeepLearningStudy//trained_model//OnyxSegmentation//");
                segmentation.Run(resizedInputImage.Data, resized_output_probability.Data, 512, 512, 1, 1);
                resized_output_probability = resized_output_probability * 512;



                Cv2.Threshold(resized_output_probability, resized_output_threshold, 240, 255, OpenCvSharp.ThresholdTypes.Binary);


                Cv2.NamedWindow("original", WindowFlags.FreeRatio);
                Cv2.NamedWindow("original_resized", WindowFlags.FreeRatio);
                Cv2.NamedWindow("resized_output_threshold", WindowFlags.FreeRatio);
                Cv2.NamedWindow("resized_output_probability", WindowFlags.FreeRatio);
                Cv2.NamedWindow("resultOverlay", WindowFlags.FreeRatio);

                Cv2.ImShow("original", original);
                Cv2.ImShow("original_resized", resizedInputImage);
                Cv2.ImShow("resized_output_threshold", resized_output_threshold);
                Cv2.ImShow("resized_output_probability", resized_output_probability);


                resized_output_threshold.ConvertTo(resized8Bit_output_threshold, MatType.CV_8UC1);
     

                SimpleBlobDetector.Params param = new SimpleBlobDetector.Params();
                param.FilterByArea = true;
                param.MinArea = 10000;
                param.MaxArea = 1000000000;
                param.FilterByColor = false;
                param.FilterByInertia = false;
                param.MinThreshold = 128;
                param.MaxThreshold = 255;
                param.FilterByConvexity = false;
                param.FilterByCircularity = false;
                param.FilterByInertia = false;
               


                SimpleBlobDetector detector = SimpleBlobDetector.Create(param);

                var keyPoints = detector.Detect(resized8Bit_output_threshold);
              
 
                System.Diagnostics.Debug.WriteLine("Key Points Count = " + keyPoints.Length);
                int xOnyxLocation = 0;
                int yOnyxLocation = 0;
                int xOnyxWidth = 0;
                int yOnyxHeight = 0;


                if (keyPoints.Length > 0)
                {
                    xOnyxLocation = (int)keyPoints[0].Pt.X - 213;
                    yOnyxLocation = (int)keyPoints[0].Pt.Y - 63;
                    xOnyxWidth = 425;
                    yOnyxHeight = 125;

               
                    
                }

       

                


                double xOriginalOnyxLocation = (double)xOnyxLocation / 512.0 * 4096.0;
                double yOriginalOnyxLocation = (double)yOnyxLocation / 512.0 * 3000.0;
                double xOriginalOnyxWidth = (double)xOnyxWidth / 512.0 * 4096.0;
                double yOriginalOnyxHeight = (double)yOnyxHeight / 512.0 * 3000.0;

                Cv2.Rectangle(resultOverlay, new Rect((int)xOriginalOnyxLocation, (int)yOriginalOnyxLocation, (int)xOriginalOnyxWidth, (int)yOriginalOnyxHeight), new Scalar(0, 255, 0),10);
                Cv2.ImShow("resultOverlay", resultOverlay);


                Cv2.WaitKey();
            }
            catch (Exception e)
            {
                System.Diagnostics.Debug.WriteLine(e.Message);
            }


        }
    }
}
