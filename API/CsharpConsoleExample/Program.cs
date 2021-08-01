using System;
using System.IO;
using OpenCvSharp;



namespace CsharpConsoleExample
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                string deskTopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);

                var files = Directory.GetFiles(deskTopPath + Path.DirectorySeparatorChar + "PCB_Original", "*.*", SearchOption.AllDirectories);
                HV.V1.DEEP.Segmentation segmentation = new HV.V1.DEEP.Segmentation();
                segmentation.Import("C://Github//DeepLearningStudy//trained_model//PCBDefectSegmentation//");

                var outfilePath = deskTopPath + Path.DirectorySeparatorChar + "PCB_Original_Resize" + Path.DirectorySeparatorChar;
                var outfilePath2 = deskTopPath + Path.DirectorySeparatorChar + "PCB_Mask" + Path.DirectorySeparatorChar;
                foreach (var file in files)
                {
                     var fileName = Path.GetFileNameWithoutExtension(file);

                    var input = Cv2.ImRead(file, ImreadModes.Color);
                    Mat resizeInput = new Mat(new Size(512, 512), OpenCvSharp.MatType.CV_32FC1);
                    Cv2.Resize(input, resizeInput, new Size(512, 512));

                    resizeInput.SaveImage(outfilePath + fileName + "_resize.jpg");

                    /*

                    var output = new Mat(new Size(512, 512), OpenCvSharp.MatType.CV_8UC1);
                    var output_32 = new Mat(new Size(512, 512), OpenCvSharp.MatType.CV_32FC1);

                    segmentation.Run(resizeInput.Data, output_32.Data, 512, 512, 3, 1);

                    output_32 = output_32 * 255;

                    Cv2.Threshold(output_32, output_32, 128, 255, OpenCvSharp.ThresholdTypes.Binary);
                    output_32.ConvertTo(output, MatType.CV_8UC1);

                    var fullPath = outfilePath + fileName + "_result.jpg";

                    output.SaveImage(outfilePath + fileName + "_resize.jpg");


                    Mat resizedOutput = new Mat(new Size(input.Width, input.Height), OpenCvSharp.MatType.CV_8UC1);

                    Cv2.Resize(output, resizedOutput, new Size(input.Width, input.Height));


                    for(int width =0; width < resizedOutput.Cols; width++)
                    {
                        for(int height =0; height < resizedOutput.Rows; height++)
                        {
                            if (resizedOutput.Get<byte>(height, width) > 128)
                                input.Set<Vec3b>(height, width, new Vec3b(0,255,0));
                        }
                    }
                    var fullPath2 = outfilePath2 + fileName + "_mask.jpg";
                    input.SaveImage(fullPath2);

                   // var masked_output = input.SetTo(new Scalar(0, 255, 0), output);

//                    var fullPath2 = outfilePath2 + fileName + "_mask.jpg";
  //                  masked_output.SaveImage(fullPath2);*/

                }




            }
            catch (Exception e)
            {
                System.Diagnostics.Debug.WriteLine(e.Message);
            }


        }
    }
}
