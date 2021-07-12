using CsvHelper;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.Command;
using OnyxChassisLocationEstimator.Model;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace OnyxChassisLocationEstimator.ViewModel
{
    public class MainWindowViewModel : ViewModelBase
    {


        private readonly HV.V1.DEEP.Segmentation onyxSegmentation;
        private readonly HV.V1.DEEP.Segmentation onyxLineSegmentation;
        private readonly HV.V1.DEEP.Segmentation onyxRoundSegmentation;
        private readonly HV.V1.DEEP.Segmentation chassisLineSegmentation;
        private readonly HV.V1.DEEP.Segmentation chassisRoundSegmentation;

        public MainWindowViewModel()
        {
            var currentDirectory = System.AppDomain.CurrentDomain.BaseDirectory;

            this.onyxSegmentation = new HV.V1.DEEP.Segmentation();
            this.onyxSegmentation.Import(currentDirectory + "OnyxSegmentation//");

            this.onyxLineSegmentation = new HV.V1.DEEP.Segmentation();
            this.onyxLineSegmentation.Import(currentDirectory + "OnyxLineSegmentation//");


            this.onyxRoundSegmentation = new HV.V1.DEEP.Segmentation();
            this.onyxRoundSegmentation.Import(currentDirectory + "OnyxRoundSegmentation//");



            this.chassisLineSegmentation = new HV.V1.DEEP.Segmentation();
            this.chassisLineSegmentation.Import(currentDirectory + "ChassisLineSegmentation");


            this.chassisRoundSegmentation = new HV.V1.DEEP.Segmentation();
            this.chassisRoundSegmentation.Import(currentDirectory + "ChassisRoundSegmentation");
            

        }



        public ICommand OpenImageCommand
        {
            get => new RelayCommand(() =>
            {
                try
                {
                    var folderName =  Helper.DialogHelper.OpenFolder();

                    var files = Directory.GetFiles(folderName, "*.*", SearchOption.AllDirectories)
                                         .Where(s => s.ToUpper().EndsWith(".PNG") || 
                                                     s.ToUpper().EndsWith(".JPG") || 
                                                     s.ToUpper().EndsWith(".JPEG") || 
                                                     s.ToUpper().EndsWith(".BMP") || 
                                                     s.ToUpper().EndsWith(".TIFF"));

                    foreach(var file in files)
                    {
                        var fileName = Path.GetFileName(file);

                        var fileModel = new Model.File()
                        {
                            FileName = fileName,
                            FilePath = file
                        };

                        this.ImageFileCollection.Add(fileModel);
                    }
                }
                catch(Exception e)
                {
                    Helper.DialogHelper.ShowToastErrorMessage("이미지 열기 실패",e.Message);
                }
                

            });
        }

        public ICommand OpenResultFolderCommand
        {
            get => new RelayCommand(() =>
            {
                try
                {
                    var currentDirectory = System.AppDomain.CurrentDomain.BaseDirectory;
                    var resultFolder = currentDirectory + "Result";

                    Process.Start("explorer.exe", resultFolder);

                }
                catch(Exception e)
                {
                    Helper.DialogHelper.ShowToastErrorMessage("결과 폴더 열기 실패", e.Message);
                }

            });
        }

        public ICommand StartAnalyzeCommand
        {
            get => new RelayCommand(() =>
            {

                if (this.IsRunning == true) return;
                this.IsRunning = true;

 
                if(this.ImageFileCollection.Count == 0)
                {
                    Helper.DialogHelper.ShowToastErrorMessage("분석 실패", "파일을 로드해주세요.");
                    this.IsRunning = false;
                }
                Task.Run(() =>
                {

                    try
                    {
                        var resultList = new List<CsvResult>();


                        foreach (var imageFileInput in this.ImageFileCollection)
                        {
                            try
                            {
                                var currentDirectory = System.AppDomain.CurrentDomain.BaseDirectory;
                                var fileNameWithOutExtension = Path.GetFileNameWithoutExtension(imageFileInput.FileName);
                                var ResultFolder = currentDirectory + "Result" + Path.DirectorySeparatorChar + fileNameWithOutExtension + "_"  + DateTime.Now.ToString("yyy-MM-dd HH-mm-ss-fff") + Path.DirectorySeparatorChar;
                                var cameraResolution = this.TopCameraResolution;

                                Directory.CreateDirectory(ResultFolder);

                                Mat original = new Mat(imageFileInput.FilePath, OpenCvSharp.ImreadModes.Grayscale);
                                Mat resultOverlay = new Mat(original.Width, original.Height, MatType.CV_8UC3);
                                Mat resizedInputImage = new Mat(512, 512, MatType.CV_8UC1);
                                Mat resized_output_probability = new Mat(512, 512, MatType.CV_32FC1);
                                Mat resized_output_threshold = new Mat(512, 512, MatType.CV_32FC1);
                                Mat resized8Bit_output_threshold = new Mat(512, 512, MatType.CV_8UC1);

                                Mat resized_output_top_line_probability = new Mat(192, 1024, MatType.CV_32FC1);
                                Mat resized_output_bottom_line_probability = new Mat(192, 1024, MatType.CV_32FC1);
                              

                                Mat resized_output_top_line_threshold = new Mat(192, 1024, MatType.CV_32FC1);
                                Mat resized_output_bottom_line_threshold = new Mat(192, 1024, MatType.CV_32FC1);

                                Mat resized8Bit_output_bottom_line_threshold = new Mat(192, 1024, MatType.CV_8UC1);
                                Mat resized8Bit_output_top_line_threshold = new Mat(192, 1024, MatType.CV_8UC1);





                                Mat resized_output_top_chassis_line_probability = new Mat(192, 1024, MatType.CV_32FC1);
                                Mat resized_output_bottom_chassis_line_probability = new Mat(192, 1024, MatType.CV_32FC1);


                                Mat resized_output_top_chassis_line_threshold = new Mat(192, 1024, MatType.CV_32FC1);
                                Mat resized_output_bottom_chassis_line_threshold = new Mat(192, 1024, MatType.CV_32FC1);

                                Mat resized8Bit_output_bottom_chassis_line_threshold = new Mat(192, 1024, MatType.CV_8UC1);
                                Mat resized8Bit_output_top_chassis_line_threshold = new Mat(192, 1024, MatType.CV_8UC1);





                                Mat resized8Bit_output_bottom_line_canny = new Mat(192, 1024, MatType.CV_8UC1);
                                Mat resized8Bit_output_top_line_canny = new Mat(192, 1024, MatType.CV_8UC1);

                                Mat resized8Bit_output_bottom_chassis_line_canny = new Mat(192, 1024, MatType.CV_8UC1);
                                Mat resized8Bit_output_top_chassis_line_canny = new Mat(192, 1024, MatType.CV_8UC1);





                                Mat resized_output_left_line_probability = new Mat(256, 256, MatType.CV_32FC1);
                                Mat resized_output_right_line_probability = new Mat(256, 256, MatType.CV_32FC1);

                                Mat resized_output_left_line_threshold = new Mat(256, 256, MatType.CV_32FC1);
                                Mat resized_output_right_line_threshold = new Mat(256, 256, MatType.CV_32FC1);

                                Mat resized8Bit_output_left_line_threshold = new Mat(256, 256, MatType.CV_8UC1);
                                Mat resized8Bit_output_right_line_threshold = new Mat(256, 256, MatType.CV_8UC1);



                                Mat resized_output_left_chassis_line_probability = new Mat(256, 256, MatType.CV_32FC1);
                                Mat resized_output_right_chassis_line_probability = new Mat(256, 256, MatType.CV_32FC1);

                                Mat resized_output_left_chassis_line_threshold = new Mat(256, 256, MatType.CV_32FC1);
                                Mat resized_output_right_chassis_line_threshold = new Mat(256, 256, MatType.CV_32FC1);

                                Mat resized8Bit_output_left_chassis_line_threshold = new Mat(256, 256, MatType.CV_8UC1);
                                Mat resized8Bit_output_right_chassis_line_threshold = new Mat(256, 256, MatType.CV_8UC1);


                                Cv2.Resize(original, resizedInputImage, new OpenCvSharp.Size(512, 512));
                                Cv2.CvtColor(original, resultOverlay, ColorConversionCodes.GRAY2BGR);

                                this.onyxSegmentation.Run(resizedInputImage.Data, resized_output_probability.Data, 512, 512, 1, 1);
                                resized_output_probability = resized_output_probability * 255;

                                Cv2.Threshold(resized_output_probability, resized_output_threshold, 240, 255, OpenCvSharp.ThresholdTypes.Binary);
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


                                //Cv2.Rectangle(resultOverlay, new OpenCvSharp.Rect((int)xOriginalOnyxLocation, (int)yOriginalOnyxLocation, (int)xOriginalOnyxWidth, (int)yOriginalOnyxHeight), new Scalar(0, 255, 0), 10);


                                double xROILocation1 = (double)xOnyxLocation / 512.0 * 4096.0;
                                double yROILocation1 = (double)yOnyxLocation / 512.0 * 3000.0;
                                double widthROILocation1 = (double)xOnyxWidth / 512.0 * 4096.0;
                                double heightROILocation1 = (double)yOnyxHeight / 512.0 * 3000.0;


                                var topROI = original.SubMat(new OpenCvSharp.Rect((int)(xROILocation1 + widthROILocation1 / 2 - 500), (int)(yROILocation1 - 100), 1024, 192)).Clone();
                                var bottomROI = original.SubMat(new OpenCvSharp.Rect((int)(xROILocation1 + widthROILocation1 / 2 - 500), (int)((yROILocation1 - 100) + heightROILocation1), 1024, 192)).Clone();

                                var leftROI = original.SubMat(new OpenCvSharp.Rect((int)(xROILocation1 - 128), (int)(yROILocation1 + (yOriginalOnyxHeight - 256)/2.0), 256, 256)).Clone();
                                var rightROI = original.SubMat(new OpenCvSharp.Rect((int)(xROILocation1 - 128 + widthROILocation1), (int)(yROILocation1 + (yOriginalOnyxHeight - 256) / 2.0), 256, 256)).Clone();



                                Cv2.Rectangle(resultOverlay, new OpenCvSharp.Rect((int)(xROILocation1 + widthROILocation1 / 2 - 500), (int)(yROILocation1 - 100), 1024, 192), new Scalar(0, 0, 255), 10);
                                Cv2.Rectangle(resultOverlay, new OpenCvSharp.Rect((int)(xROILocation1 + widthROILocation1 / 2 - 500), (int)((yROILocation1 - 100) + heightROILocation1), 1024, 192), new Scalar(0, 0, 255), 10);
                                Cv2.Rectangle(resultOverlay, new OpenCvSharp.Rect((int)(xROILocation1 - 128), (int)(yROILocation1 + (yOriginalOnyxHeight - 256) / 2.0), 256, 256), new Scalar(0, 0, 255), 10);
                                Cv2.Rectangle(resultOverlay, new OpenCvSharp.Rect((int)(xROILocation1 - 128 + widthROILocation1), (int)(yROILocation1 + (yOriginalOnyxHeight - 256) / 2.0), 256, 256), new Scalar(0, 0, 255), 10);



                                var OnyxArea512PredictionPath = ResultFolder + "OnyxArea_Prediction_512.jpg";
                                var OnyxArea512ThresholdPath = ResultFolder + "OnyxArea_Threshold_512.jpg";
                                var OriginalResizePath = ResultFolder + "Original_512.jpg";
                                //var OverlayResult = ResultFolder + "Overlay.jpg";

                                var OverlayResultPath = ResultFolder + "Overlay.jpg";


                                var DateTimeNow = DateTime.Now.ToString("yyy-MM-dd HH-ss-fff");

                                var TopROIPath = ResultFolder + DateTimeNow + "TopLineROI.jpg";
                                var BottomROIPath = ResultFolder + DateTimeNow + "BottomLineROI.jpg";

                                var LeftROIPath = ResultFolder + DateTimeNow + "LeftLineROI.jpg";
                                var RightROIPath = ResultFolder + DateTimeNow + "RightLineROI.jpg";


                                var TopROIOnyxPath = ResultFolder + DateTimeNow + "TopLineOnyxROI.jpg";
                                var BottomROIOnyxPath = ResultFolder + DateTimeNow + "BottomLineOnyxROI.jpg";



                                var TopROIChassisPath = ResultFolder + DateTimeNow + "TopLineChassisROI.jpg";
                                var BottomROIChassisPath = ResultFolder + DateTimeNow + "BottomLineChassisROI.jpg";

                                var LeftROIOnyxPath = ResultFolder + DateTimeNow + "LeftLineOnyxROI.jpg";
                                var RightROIOnyxPath = ResultFolder + DateTimeNow + "RightLineOnyxROI.jpg";


                                var LeftROIChassisPath = ResultFolder + DateTimeNow + "LeftLineChassisROI.jpg";
                                var RightROIChassisPath = ResultFolder + DateTimeNow + "RightLineChassisROI.jpg";


                                var TopROIOnyxCannyPath = ResultFolder + DateTimeNow + "TopLineOnyxCannyROI.jpg";
                                var BottomROIOnyxCannyPath = ResultFolder + DateTimeNow + "BottomLineOnyxCannyROI.jpg";
                                var TopROIChassisCannyPath = ResultFolder + DateTimeNow + "TopLineChassisCannyROI.jpg";
                                var BottomROIChassisCannyPath = ResultFolder + DateTimeNow + "BottomLineChassisCannyROI.jpg";




                                this.onyxLineSegmentation.Run(topROI.Data, resized_output_top_line_probability.Data, 1024, 192, 1, 1);
                                resized_output_top_line_probability = resized_output_top_line_probability * 255;
                                Cv2.Threshold(resized_output_top_line_probability, resized_output_top_line_threshold, 178, 255, OpenCvSharp.ThresholdTypes.Binary);
                                resized_output_top_line_threshold.ConvertTo(resized8Bit_output_top_line_threshold, MatType.CV_8UC1);



                                this.onyxLineSegmentation.Run(bottomROI.Data, resized_output_bottom_line_probability.Data, 1024, 192, 1, 1);
                                resized_output_bottom_line_probability = resized_output_bottom_line_probability * 255;
                                Cv2.Threshold(resized_output_bottom_line_probability, resized_output_bottom_line_threshold, 178, 255, OpenCvSharp.ThresholdTypes.Binary);
                                resized_output_bottom_line_threshold.ConvertTo(resized8Bit_output_bottom_line_threshold, MatType.CV_8UC1);



                                this.onyxRoundSegmentation.Run(leftROI.Data, resized_output_left_line_probability.Data, 256, 256, 1, 1);
                                resized_output_left_line_probability = resized_output_left_line_probability * 255;
                                Cv2.Threshold(resized_output_left_line_probability, resized_output_left_line_threshold, 204, 255, OpenCvSharp.ThresholdTypes.Binary);
                                resized_output_left_line_threshold.ConvertTo(resized8Bit_output_left_line_threshold, MatType.CV_8UC1);



                                this.onyxRoundSegmentation.Run(rightROI.Data, resized_output_right_line_probability.Data, 256, 256, 1, 1);
                                resized_output_right_line_probability = resized_output_right_line_probability * 255;
                                Cv2.Threshold(resized_output_right_line_probability, resized_output_right_line_threshold, 204, 255, OpenCvSharp.ThresholdTypes.Binary);
                                resized_output_right_line_threshold.ConvertTo(resized8Bit_output_right_line_threshold, MatType.CV_8UC1);
                                // Onyx line segmentation 




                                this.chassisLineSegmentation.Run(topROI.Data, resized_output_top_chassis_line_probability.Data, 1024, 192, 1, 1);
                                resized_output_top_chassis_line_probability = resized_output_top_chassis_line_probability * 255;
                                Cv2.Threshold(resized_output_top_chassis_line_probability, resized_output_top_chassis_line_threshold, 204, 255, OpenCvSharp.ThresholdTypes.Binary);
                                resized_output_top_chassis_line_threshold.ConvertTo(resized8Bit_output_top_chassis_line_threshold, MatType.CV_8UC1);



                                this.chassisLineSegmentation.Run(bottomROI.Data, resized_output_bottom_chassis_line_probability.Data, 1024, 192, 1, 1);
                                resized_output_bottom_chassis_line_probability = resized_output_bottom_chassis_line_probability * 255;
                                Cv2.Threshold(resized_output_bottom_chassis_line_probability, resized_output_bottom_chassis_line_threshold, 204, 255, OpenCvSharp.ThresholdTypes.Binary);
                                resized_output_bottom_chassis_line_threshold.ConvertTo(resized8Bit_output_bottom_chassis_line_threshold, MatType.CV_8UC1);


                                this.chassisRoundSegmentation.Run(leftROI.Data, resized_output_left_chassis_line_probability.Data, 256, 256, 1, 1);
                                resized_output_left_chassis_line_probability = resized_output_left_chassis_line_probability * 255;
                                Cv2.Threshold(resized_output_left_chassis_line_probability, resized_output_left_chassis_line_threshold, 204, 255, OpenCvSharp.ThresholdTypes.Binary);
                                resized_output_left_chassis_line_threshold.ConvertTo(resized8Bit_output_left_chassis_line_threshold, MatType.CV_8UC1);


                                this.chassisRoundSegmentation.Run(rightROI.Data, resized_output_right_chassis_line_probability.Data, 256, 256, 1, 1);
                                resized_output_right_chassis_line_probability = resized_output_right_chassis_line_probability * 255;
                                Cv2.Threshold(resized_output_right_chassis_line_probability, resized_output_right_chassis_line_threshold, 204, 255, OpenCvSharp.ThresholdTypes.Binary);
                                resized_output_right_chassis_line_threshold.ConvertTo(resized8Bit_output_right_chassis_line_threshold, MatType.CV_8UC1);
                                // Chassis line segmentation 




                                //var cannyTopOnyxLine = resized8Bit_output_top_line_threshold.Canny(0, 128);
                                //var cannyBottomOnyxLine = resized8Bit_output_bottom_line_threshold.Canny(0, 128);



                                //var cannyTopChassisLine = resized8Bit_output_top_chassis_line_threshold.Canny(0, 128);
                                //var cannyBottomChassisLine = resized8Bit_output_bottom_chassis_line_threshold.Canny(0, 128);




                                var topROILocationX = (int)(xROILocation1 + widthROILocation1 / 2 - 500);
                                var topROILocationY = (int)(yROILocation1 - 100);


                                var bottomROILocationX = (int)(xROILocation1 + widthROILocation1 / 2 - 500);
                                var bottomROILocationY = (int)((yROILocation1 - 100) + heightROILocation1);


                    
                                var leftROILocationX = (int)(xROILocation1 - 128);
                                var leftROILocationY = (int)(yROILocation1 + (yOriginalOnyxHeight - 256) / 2.0);

                                var rightROILocationX = (int)(xROILocation1 - 128 + widthROILocation1);
                                var rightROILocationY = (int)(yROILocation1 + (yOriginalOnyxHeight - 256) / 2.0);




                                ///Onyx Corner Calculation
                                var algorithm = new HV.V1.ALGORITHM.Algorithm();
                                //Top Line
                                var topOnyxLineInfo = new HV.V1.ALGORITHM.MeasurementLine();
                                topOnyxLineInfo.CenterX = 512;
                                topOnyxLineInfo.CenterY = 96;
                                topOnyxLineInfo.Angle = 0;
                                topOnyxLineInfo.Range = 800;
                                topOnyxLineInfo.Distance = 192;
                                topOnyxLineInfo.Type = 0;
                                topOnyxLineInfo.Threadhold = 2;
                                topOnyxLineInfo.Direction = true;

                                HV.V1.ALGORITHM.Line topOnyxLine;
                                if (algorithm.FindLine(resized8Bit_output_top_line_threshold.Data, 1024, 192, topOnyxLineInfo, out topOnyxLine) == false)
                                {
                                    throw new Exception("Top Line Detection failed");
                                }
                                topOnyxLine.CX += topROILocationX;
                                topOnyxLine.CY += topROILocationY;
                                //topOnyxLine.Theta = algorithm.GetAlignmentDegreeFromDegree(topOnyxLine.Theta, 0);


                                //Bottom Line
                                var bottomOnyxLineInfo = new HV.V1.ALGORITHM.MeasurementLine();
                                bottomOnyxLineInfo.CenterX = 512;
                                bottomOnyxLineInfo.CenterY = 96;
                                bottomOnyxLineInfo.Angle = 0;
                                bottomOnyxLineInfo.Range = 800;
                                bottomOnyxLineInfo.Distance = 192;
                                bottomOnyxLineInfo.Type = 0;
                                bottomOnyxLineInfo.Threadhold = 2;
                                bottomOnyxLineInfo.Direction = false;

                                HV.V1.ALGORITHM.Line bottomOnyxLine;
                                if(algorithm.FindLine(resized8Bit_output_bottom_line_threshold.Data, 1024, 192, bottomOnyxLineInfo, out bottomOnyxLine) == false)
                                {
                                    throw new Exception("Bottom Line Detection failed");
                                }
                                bottomOnyxLine.CX += bottomROILocationX;
                                bottomOnyxLine.CY += bottomROILocationY;
                                //bottomOnyxLine.Theta = algorithm.GetAlignmentDegreeFromDegree(bottomOnyxLine.Theta, 0);

                                var averageHorizontalAngle = (bottomOnyxLine.Theta + topOnyxLine.Theta) / 2;


                                //Left Line
                                var leftOnyxLineInfo = new HV.V1.ALGORITHM.MeasurementLine();
                                leftOnyxLineInfo.CenterX = 128;
                                leftOnyxLineInfo.CenterY = 128;
                                leftOnyxLineInfo.Angle = 90;
                                leftOnyxLineInfo.Range = 35;
                                leftOnyxLineInfo.Distance = 256;
                                leftOnyxLineInfo.Type = 0;
                                leftOnyxLineInfo.Threadhold = 2;
                                leftOnyxLineInfo.Direction = false;

                                HV.V1.ALGORITHM.Line leftOnyxLine;
                                if (algorithm.FindLine(resized8Bit_output_left_line_threshold.Data, 256, 256, leftOnyxLineInfo, out leftOnyxLine) == false)
                                {
                                    throw new Exception("Left Line Detection failed");
                                }
                                leftOnyxLine.CX += leftROILocationX;
                                leftOnyxLine.CY += leftROILocationY;
                                leftOnyxLine.Theta = (averageHorizontalAngle + 90);




                                //Right Line
                                var rightOnyxLineInfo = new HV.V1.ALGORITHM.MeasurementLine();
                                rightOnyxLineInfo.CenterX = 128;
                                rightOnyxLineInfo.CenterY = 128;
                                rightOnyxLineInfo.Angle = 90;
                                rightOnyxLineInfo.Range = 35;
                                rightOnyxLineInfo.Distance = 256;
                                rightOnyxLineInfo.Type = 0;
                                rightOnyxLineInfo.Threadhold = 2;
                                rightOnyxLineInfo.Direction = true;

                                HV.V1.ALGORITHM.Line rightOnyxLine;
                                if (algorithm.FindLine(resized8Bit_output_right_line_threshold.Data, 256, 256, rightOnyxLineInfo, out rightOnyxLine) == false)
                                {
                                    throw new Exception("Right Line Detection failed");
                                }
                                rightOnyxLine.CX += rightROILocationX;
                                rightOnyxLine.CY += rightROILocationY;
                                rightOnyxLine.Theta = (averageHorizontalAngle + 90);


                                double leftOnyxTopX = 0;
                                double leftOnyxTopY = 0;
                                double rightOnyxTopX = 0;
                                double rightOnyxTopY = 0;
                                double leftOnyxBottomX = 0;
                                double leftOnyxBottomY = 0;
                                double rightOnyxBottomX = 0;
                                double rightOnyxBottomY = 0;

                                algorithm.GetCrossPointFromTwoLine(leftOnyxLine, topOnyxLine, out leftOnyxTopX, out leftOnyxTopY);
                                algorithm.GetCrossPointFromTwoLine(rightOnyxLine, topOnyxLine, out rightOnyxTopX, out rightOnyxTopY);
                                algorithm.GetCrossPointFromTwoLine(leftOnyxLine, bottomOnyxLine, out leftOnyxBottomX, out leftOnyxBottomY);
                                algorithm.GetCrossPointFromTwoLine(rightOnyxLine, bottomOnyxLine, out rightOnyxBottomX, out rightOnyxBottomY);

                                ///Onyx Corner Calculation
                                ///




                                /// Chassis Corner Calculation
                                /// 

                                //var algorithm = new HV.V1.ALGORITHM.Algorithm();
                                //Top Line
                                var topChassisLineInfo = new HV.V1.ALGORITHM.MeasurementLine();
                                topChassisLineInfo.CenterX = 512;
                                topChassisLineInfo.CenterY = 96;
                                topChassisLineInfo.Angle = 0;
                                topChassisLineInfo.Range = 800;
                                topChassisLineInfo.Distance = 192;
                                topChassisLineInfo.Type = 0;
                                topChassisLineInfo.Threadhold = 2;
                                topChassisLineInfo.Direction = true;

                                HV.V1.ALGORITHM.Line topChassisLine;
                                if (algorithm.FindLine(resized8Bit_output_top_chassis_line_threshold.Data, 1024, 192, topChassisLineInfo, out topChassisLine) == false)
                                {
                                    throw new Exception("Top Line Detection failed");
                                }
                                topChassisLine.CX += topROILocationX;
                                topChassisLine.CY += topROILocationY;
                                //topOnyxLine.Theta = algorithm.GetAlignmentDegreeFromDegree(topOnyxLine.Theta, 0);


                                //Bottom Line
                                var bottomChassisLineInfo = new HV.V1.ALGORITHM.MeasurementLine();
                                bottomChassisLineInfo.CenterX = 512;
                                bottomChassisLineInfo.CenterY = 96;
                                bottomChassisLineInfo.Angle = 0;
                                bottomChassisLineInfo.Range = 800;
                                bottomChassisLineInfo.Distance = 192;
                                bottomChassisLineInfo.Type = 0;
                                bottomChassisLineInfo.Threadhold = 2;
                                bottomChassisLineInfo.Direction = false;

                                HV.V1.ALGORITHM.Line bottomChassisLine;
                                if (algorithm.FindLine(resized8Bit_output_bottom_chassis_line_threshold.Data, 1024, 192, bottomChassisLineInfo, out bottomChassisLine) == false)
                                {
                                    throw new Exception("Bottom Line Detection failed");
                                }
                                bottomChassisLine.CX += bottomROILocationX;
                                bottomChassisLine.CY += bottomROILocationY;
                                //bottomOnyxLine.Theta = algorithm.GetAlignmentDegreeFromDegree(bottomOnyxLine.Theta, 0);

                                var averageChassisHorizontalAngle = (bottomChassisLine.Theta + topChassisLine.Theta) / 2;


                                //Left Line
                                var leftChassisLineInfo = new HV.V1.ALGORITHM.MeasurementLine();
                                leftChassisLineInfo.CenterX = 128;
                                leftChassisLineInfo.CenterY = 128;
                                leftChassisLineInfo.Angle = 90;
                                leftChassisLineInfo.Range = 35;
                                leftChassisLineInfo.Distance = 256;
                                leftChassisLineInfo.Type = 0;
                                leftChassisLineInfo.Threadhold = 2;
                                leftChassisLineInfo.Direction = false;

                                HV.V1.ALGORITHM.Line leftChassisLine;
                                if (algorithm.FindLine(resized8Bit_output_left_chassis_line_threshold.Data, 256, 256, leftChassisLineInfo, out leftChassisLine) == false)
                                {
                                    throw new Exception("Left Line Detection failed");
                                }
                                leftChassisLine.CX += leftROILocationX;
                                leftChassisLine.CY += leftROILocationY;
                                leftChassisLine.Theta = (averageChassisHorizontalAngle + 90);




                                //Right Line
                                var rightChassisLineInfo = new HV.V1.ALGORITHM.MeasurementLine();
                                rightChassisLineInfo.CenterX = 128;
                                rightChassisLineInfo.CenterY = 128;
                                rightChassisLineInfo.Angle = 90;
                                rightChassisLineInfo.Range = 35;
                                rightChassisLineInfo.Distance = 256;
                                rightChassisLineInfo.Type = 0;
                                rightChassisLineInfo.Threadhold = 2;
                                rightChassisLineInfo.Direction = true;

                                HV.V1.ALGORITHM.Line rightChassisLine;
                                if (algorithm.FindLine(resized8Bit_output_right_chassis_line_threshold.Data, 256, 256, rightChassisLineInfo, out rightChassisLine) == false)
                                {
                                    throw new Exception("Right Line Detection failed");
                                }
                                rightChassisLine.CX += rightROILocationX;
                                rightChassisLine.CY += rightROILocationY;
                                rightChassisLine.Theta = (averageChassisHorizontalAngle + 90);


                                double leftChassisTopX = 0;
                                double leftChassisTopY = 0;
                                double rightChassisTopX = 0;
                                double rightChassisTopY = 0;
                                double leftChassisBottomX = 0;
                                double leftChassisBottomY = 0;
                                double rightChassisBottomX = 0;
                                double rightChassisBottomY = 0;

                                algorithm.GetCrossPointFromTwoLine(leftChassisLine, topChassisLine, out leftChassisTopX, out leftChassisTopY);
                                algorithm.GetCrossPointFromTwoLine(rightChassisLine, topChassisLine, out rightChassisTopX, out rightChassisTopY);
                                algorithm.GetCrossPointFromTwoLine(leftChassisLine, bottomChassisLine, out leftChassisBottomX, out leftChassisBottomY);
                                algorithm.GetCrossPointFromTwoLine(rightChassisLine, bottomChassisLine, out rightChassisBottomX, out rightChassisBottomY);
                                /// Chassis Corner Calculation





                                /// Onyx Center X,Y,R Calculation
                                /// 
                                var TopOnyxCenterX = (leftOnyxTopX + rightOnyxTopX) / 2;
                                var TopOnyxCenterY = (leftOnyxTopY + rightOnyxTopY) / 2;

                                var BottomOnyxCenterX = (leftOnyxBottomX + rightOnyxBottomX) / 2;
                                var BottomOnyxCenterY = (leftOnyxBottomY + rightOnyxBottomY) / 2;


                                var LeftOnyxCenterX = (leftOnyxTopX + leftOnyxBottomX) / 2;
                                var LeftOnyxCenterY = (leftOnyxTopY + leftOnyxBottomY) / 2;


                                var RightOnyxCenterX = (rightOnyxTopX + rightOnyxBottomX) / 2;
                                var RightOnyxCenterY = (rightOnyxTopY + rightOnyxBottomY) / 2;


                                var OnyxHorizontalDistance = Math.Sqrt(Math.Pow((LeftOnyxCenterX - RightOnyxCenterX), 2) + Math.Pow((LeftOnyxCenterY - RightOnyxCenterY), 2));
                                var OnyxVerticalDistance = Math.Sqrt(Math.Pow((TopOnyxCenterX - BottomOnyxCenterX), 2) + Math.Pow((TopOnyxCenterY - BottomOnyxCenterY), 2));

                                var OnyxCenterX = (leftOnyxTopX + rightOnyxTopX + leftOnyxBottomX + rightOnyxBottomX) / 4;
                                var OnyxCenterY = (leftOnyxTopY + rightOnyxTopY + leftOnyxBottomY + rightOnyxBottomY) / 4;

                                double OnyxTiltAngle = Math.Atan2(RightOnyxCenterX - LeftOnyxCenterX, RightOnyxCenterY - LeftOnyxCenterY) * 180.0 / Math.PI - 90;



                                /// Chassis Center X,Y,R Calculation
                                /// 
                                var TopChassisCenterX = (leftChassisTopX + rightChassisTopX) / 2;
                                var TopChassisCenterY = (leftChassisTopY + rightChassisTopY) / 2;

                                var BottomChassisCenterX = (leftChassisBottomX + rightChassisBottomX) / 2;
                                var BottomChassisCenterY = (leftChassisBottomY + rightChassisBottomY) / 2;


                                var LeftChassisCenterX = (leftChassisTopX + leftChassisBottomX) / 2;
                                var LeftChassisCenterY = (leftChassisTopY + leftChassisBottomY) / 2;


                                var RightChassisCenterX = (rightChassisTopX + rightChassisBottomX) / 2;
                                var RightChassisCenterY = (rightChassisTopY + rightChassisBottomY) / 2;


                                var ChassisHorizontalDistance = Math.Sqrt(Math.Pow((LeftChassisCenterX - RightChassisCenterX), 2) + Math.Pow((LeftChassisCenterY - RightChassisCenterY), 2));
                                var ChassisVerticalDistance = Math.Sqrt(Math.Pow((TopChassisCenterX - BottomChassisCenterX), 2) + Math.Pow((TopChassisCenterY - BottomChassisCenterY), 2));


                                var ChassisCenterX = (leftChassisTopX + rightChassisTopX + leftChassisBottomX + rightChassisBottomX) / 4;
                                var ChassisCenterY = (leftChassisTopY + rightChassisTopY + leftChassisBottomY + rightChassisBottomY) / 4;

                                double ChassisTiltAngle = Math.Atan2(RightChassisCenterX - LeftChassisCenterX, RightChassisCenterY - LeftChassisCenterY) * 180.0 / Math.PI - 90;



                                // Onyx Line Draw
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(leftOnyxTopX, leftOnyxTopY), new Scalar(255, 0, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(rightOnyxTopX, rightOnyxTopY), new Scalar(255, 0, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(leftOnyxBottomX, leftOnyxBottomY), new Scalar(255, 0, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(rightOnyxBottomX, rightOnyxBottomY), new Scalar(255, 0, 0), MarkerTypes.Cross, 50, 2);


                                resultOverlay.Line(new OpenCvSharp.Point(leftOnyxTopX, leftOnyxTopY), new OpenCvSharp.Point(rightOnyxTopX, rightOnyxTopY), new Scalar(255, 0, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(leftOnyxTopX, leftOnyxTopY), new OpenCvSharp.Point(leftOnyxBottomX, leftOnyxBottomY), new Scalar(255, 0, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(rightOnyxTopX, rightOnyxTopY), new OpenCvSharp.Point(rightOnyxBottomX, rightOnyxBottomY), new Scalar(255, 0, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(leftOnyxBottomX, leftOnyxBottomY), new OpenCvSharp.Point(rightOnyxBottomX, rightOnyxBottomY), new Scalar(255, 0, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(LeftOnyxCenterX, LeftOnyxCenterY), new OpenCvSharp.Point(RightOnyxCenterX, RightOnyxCenterY), new Scalar(255, 0, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(TopOnyxCenterX, TopOnyxCenterY), new OpenCvSharp.Point(BottomOnyxCenterX, BottomOnyxCenterY), new Scalar(255, 0, 0), 2);


                                resultOverlay.DrawMarker(new OpenCvSharp.Point(topOnyxLine.CX, topOnyxLine.CY), new Scalar(255,0,0),MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(bottomOnyxLine.CX, bottomOnyxLine.CY), new Scalar(255, 0, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(leftOnyxLine.CX, leftOnyxLine.CY), new Scalar(255, 0, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(rightOnyxLine.CX, rightOnyxLine.CY), new Scalar(255, 0, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(OnyxCenterX, OnyxCenterY), new Scalar(255, 0, 0), MarkerTypes.Cross, 50, 4);

                                resultOverlay.PutText("OnyxX = " + OnyxCenterX * cameraResolution, new OpenCvSharp.Point(100, 100), HersheyFonts.HersheyPlain, 4, new Scalar(255, 0, 0), 5);
                                resultOverlay.PutText("OnyxY = " + OnyxCenterY * cameraResolution, new OpenCvSharp.Point(100, 150), HersheyFonts.HersheyPlain, 4, new Scalar(255, 0, 0), 5);
                                resultOverlay.PutText("OnyxAngle = " + OnyxTiltAngle, new OpenCvSharp.Point(100, 200), HersheyFonts.HersheyPlain, 4, new Scalar(255, 0, 0), 5);
                                resultOverlay.PutText("OnyxHorizontalDistance = " + OnyxHorizontalDistance * cameraResolution, new OpenCvSharp.Point(100, 250), HersheyFonts.HersheyPlain, 4, new Scalar(255, 0, 0), 5);
                                resultOverlay.PutText("OnyxVerticalDistance = " + OnyxVerticalDistance * cameraResolution, new OpenCvSharp.Point(100, 300), HersheyFonts.HersheyPlain, 4, new Scalar(255, 0, 0), 5);


                                // Chassis Line Draw
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(leftChassisTopX, leftChassisTopY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(rightChassisTopX, rightChassisTopY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(leftChassisBottomX, leftChassisBottomY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(rightChassisBottomX, rightChassisBottomY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 2);


                                resultOverlay.Line(new OpenCvSharp.Point(leftChassisTopX, leftChassisTopY), new OpenCvSharp.Point(rightChassisTopX, rightChassisTopY), new Scalar(0, 255, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(leftChassisTopX, leftChassisTopY), new OpenCvSharp.Point(leftChassisBottomX, leftChassisBottomY), new Scalar(0, 255, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(rightChassisTopX, rightChassisTopY), new OpenCvSharp.Point(rightChassisBottomX, rightChassisBottomY), new Scalar(0, 255, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(leftChassisBottomX, leftChassisBottomY), new OpenCvSharp.Point(rightChassisBottomX, rightChassisBottomY), new Scalar(0, 255, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(LeftChassisCenterX, LeftChassisCenterY), new OpenCvSharp.Point(RightChassisCenterX, RightChassisCenterY), new Scalar(0, 255, 0), 2);
                                resultOverlay.Line(new OpenCvSharp.Point(TopChassisCenterX, TopChassisCenterY), new OpenCvSharp.Point(BottomChassisCenterX, BottomChassisCenterY), new Scalar(0, 255, 0), 2);


                                resultOverlay.DrawMarker(new OpenCvSharp.Point(topChassisLine.CX, topOnyxLine.CY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(bottomChassisLine.CX, bottomOnyxLine.CY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(leftChassisLine.CX, leftChassisLine.CY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(rightChassisLine.CX, rightChassisLine.CY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 2);
                                resultOverlay.DrawMarker(new OpenCvSharp.Point(ChassisCenterX, ChassisCenterY), new Scalar(0, 255, 0), MarkerTypes.Cross, 50, 4);


                                resultOverlay.PutText("ChassisX = " + ChassisCenterX * cameraResolution, new OpenCvSharp.Point(100, 350), HersheyFonts.HersheyPlain, 4, new Scalar(0, 255, 0), 5);
                                resultOverlay.PutText("ChassisY = " + ChassisCenterY * cameraResolution, new OpenCvSharp.Point(100, 400), HersheyFonts.HersheyPlain, 4, new Scalar(0, 255, 0), 5);
                                resultOverlay.PutText("ChassisAngle = " + ChassisTiltAngle, new OpenCvSharp.Point(100, 450), HersheyFonts.HersheyPlain, 4, new Scalar(0, 255, 0), 5);
                                resultOverlay.PutText("ChassisHorizontalDistance = " + ChassisHorizontalDistance * cameraResolution, new OpenCvSharp.Point(100, 500), HersheyFonts.HersheyPlain, 4, new Scalar(0, 255, 0), 5);
                                resultOverlay.PutText("ChassisVerticalDistance = " + ChassisVerticalDistance * cameraResolution, new OpenCvSharp.Point(100, 550), HersheyFonts.HersheyPlain, 4, new Scalar(0, 255, 0), 5);








                                resizedInputImage.SaveImage(OriginalResizePath);
                                resized_output_probability.SaveImage(OnyxArea512PredictionPath);
                                resized_output_threshold.SaveImage(OnyxArea512ThresholdPath);


                           

                                resized8Bit_output_left_line_threshold.SaveImage(LeftROIOnyxPath);
                                resized8Bit_output_right_line_threshold.SaveImage(RightROIOnyxPath);
                                resized8Bit_output_left_chassis_line_threshold.SaveImage(LeftROIChassisPath);
                                resized8Bit_output_right_chassis_line_threshold.SaveImage(RightROIChassisPath);


                                resultOverlay.SaveImage(OverlayResultPath);


                                //ROI
                                topROI.SaveImage(TopROIPath);
                                bottomROI.SaveImage(BottomROIPath);
                                leftROI.SaveImage(LeftROIPath);
                                rightROI.SaveImage(RightROIPath);



                                resized8Bit_output_top_line_threshold.SaveImage(TopROIOnyxPath);
                                resized8Bit_output_bottom_line_threshold.SaveImage(BottomROIOnyxPath);

                                //resized8Bit_output_bottom_chassis_line_threshold.SaveImage(BottomROIChassisCannyPath);
                                //resized8Bit_output_top_chassis_line_threshold.SaveImage(TopROIChassisCannyPath);
                                



                                //ROI Deep Learning Result
                                resized_output_bottom_line_probability.Dispose();
                                resized_output_bottom_line_threshold.Dispose();
                                resized_output_top_line_probability.Dispose();
                                resized_output_top_line_threshold.Dispose();
                                resized_output_left_line_probability.Dispose();
                                resized_output_right_line_probability.Dispose();
                                resized_output_left_line_threshold.Dispose();
                                resized_output_right_line_threshold.Dispose();
                                resized8Bit_output_top_line_threshold.Dispose();
                                resized8Bit_output_bottom_line_threshold.Dispose();
                                resized8Bit_output_left_line_threshold.Dispose();
                                resized8Bit_output_right_line_threshold.Dispose();

                                resized_output_top_chassis_line_probability.Dispose();
                                resized_output_top_chassis_line_threshold.Dispose();
                                resized8Bit_output_top_chassis_line_threshold.Dispose();



                                resized_output_bottom_chassis_line_probability.Dispose();
                                resized_output_bottom_chassis_line_threshold.Dispose();
                                resized8Bit_output_bottom_chassis_line_threshold.Dispose();



                                original.Dispose();
                                resultOverlay.Dispose();
                                resizedInputImage.Dispose();
                                resized_output_probability.Dispose();
                                resized_output_threshold.Dispose();
                                resized8Bit_output_threshold.Dispose();

                                //ROI
                                topROI.Dispose();
                                bottomROI.Dispose();
                                leftROI.Dispose();
                                rightROI.Dispose();


                                resultList.Add(new CsvResult()
                                {
                                    FileName = imageFileInput.FilePath,
                                    OnyxAngle = OnyxTiltAngle,
                                    OnyxLocationPixelX = OnyxCenterX,
                                    OnyxLocationPixelY = OnyxCenterY,
                                    OnyxLocationUmX = OnyxCenterX * cameraResolution,
                                    OnyxLocationUmY = OnyxCenterY * cameraResolution,
                                    OnyxUmHorizontalDisitance = OnyxHorizontalDistance * cameraResolution,
                                    OnyxUmVerticalDisitance = OnyxVerticalDistance * cameraResolution,

                                    ChassisAngle = ChassisTiltAngle,
                                    ChassisLocationPixelX = ChassisCenterX,
                                    ChassisLocationPixelY = ChassisCenterY,
                                    ChassisLocationUmX = ChassisCenterX * cameraResolution,
                                    ChassisLocationUmY = ChassisCenterY * cameraResolution,
                                    ChassisUmHorizontalDisitance = ChassisHorizontalDistance * cameraResolution,
                                    ChassisUmVerticalDisitance = ChassisVerticalDistance * cameraResolution
                                }) ;
           

                            }
                            catch (Exception e)
                            {
                                System.Diagnostics.Debug.WriteLine(e.Message);
                                Helper.DialogHelper.ShowToastErrorMessage("단일 파일 분석 실패", e.Message);
                            }

                        }
                        var totalCsvFileName = System.AppDomain.CurrentDomain.BaseDirectory + "Result" + Path.DirectorySeparatorChar  +  "Total" + DateTime.Now.ToString("yyy-MM-dd HH-mm-ss-fff") + ".csv";
                        using (var writer = new StreamWriter(totalCsvFileName))
                        using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
                        {
                            csv.WriteRecords(resultList);
                        }

                    }
                    catch(Exception e)
                    {
                        System.Diagnostics.Debug.WriteLine(e.Message);
                        Helper.DialogHelper.ShowToastErrorMessage("분석 실패", e.Message);
                    }

          

                    this.IsRunning = false;
                });

            });
        }


        private bool _IsRunning = false;
        public bool IsRunning
        {
            get => _IsRunning;
            set => Set(ref _IsRunning, value);
        }


        private double _TopCameraResolution = 0.004928;
        public double TopCameraResolution
        {
            get => _TopCameraResolution;
            set => Set(ref _TopCameraResolution, value);
        }


        private ObservableCollection<Model.File> _ImageFileCollection = null;
        public ObservableCollection<Model.File> ImageFileCollection
        {
            get
            {
                _ImageFileCollection ??= new ObservableCollection<Model.File>();
                return _ImageFileCollection;
            }
        }


        private ObservableCollection<Model.DrawObject> _ResultDrawObjectCollection = null;
        public ObservableCollection<Model.DrawObject> ResultDrawObjectCollection
        {
            get => _ResultDrawObjectCollection;
            set => Set(ref _ResultDrawObjectCollection, value);
        }


        private WriteableBitmap _CurrentImage = null;
        public WriteableBitmap CurrentImage
        {
            get => _CurrentImage;
            set => Set(ref _CurrentImage, value);
        }


        private Model.File _SelectedImageFile = null;
        public Model.File SelectedImageFile
        {
            get => _SelectedImageFile;
            set{
                if(value != null)
                {
                    Model.File file = value;
                    if(System.IO.File.Exists(file.FilePath) == true)
                    {
                        try
                        {
                            Mat image = new Mat(file.FilePath, ImreadModes.Grayscale);
                            
                            CurrentImage = OpenCvSharp.WpfExtensions.WriteableBitmapConverter.ToWriteableBitmap(image);
                            ResultDrawObjectCollection = file.ResultCollection;
                        }
                        catch(Exception e)
                        {
                            System.Diagnostics.Debug.WriteLine(e.Message);
                        }
                        

                    }
                }


                Set(ref _SelectedImageFile, value);
            }
        }
    }
}
