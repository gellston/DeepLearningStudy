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

                var files = Directory.GetFiles(deskTopPath + Path.DirectorySeparatorChar + "PCB_Resize_original", "*.*", SearchOption.AllDirectories);
                HV.V1.DEEP.Segmentation segmentation = new HV.V1.DEEP.Segmentation();
                segmentation.Import("C://Github//DeepLearningStudy//trained_model//PCBDefectSegmentation//");

                var outfilePath = deskTopPath + Path.DirectorySeparatorChar + "PCB_Result" + Path.DirectorySeparatorChar;
                foreach (var file in files)
                {
                    var fileName = Path.GetFileNameWithoutExtension(file);

                    var input = Cv2.ImRead(file, ImreadModes.Color);
                    var input_32 = new Mat(new Size(1024, 1024), OpenCvSharp.MatType.CV_32FC3);
                    input.ConvertTo(input_32, MatType.CV_32FC3);

                    var output = new Mat(new Size(1024, 1024), OpenCvSharp.MatType.CV_8UC1);
                    var output_32 = new Mat(new Size(1024, 1024), OpenCvSharp.MatType.CV_32FC1);

                    segmentation.Run(input.Data, output_32.Data, 1024, 1024, 3, 1);

                    output_32 = output_32 * 255;

                    Cv2.Threshold(output_32, output_32, 128, 255, OpenCvSharp.ThresholdTypes.Binary);
                    output_32.ConvertTo(output, MatType.CV_8UC1);

                    var fullPath = outfilePath + fileName + "_result.jpg";

                    output.SaveImage(fullPath);
                    
                }
               
                
               
               
            }
            catch (Exception e)
            {
                System.Diagnostics.Debug.WriteLine(e.Message);
            }


        }
    }
}
