using System;
using System.Collections.Generic;
using System.Text;

namespace OnyxChassisLocationEstimator.Model
{
    public class CsvResult
    {

        public CsvResult()
        {

        }

        public string FileName { get; set; }

        public double OnyxLocationPixelX { get; set; }
        public double OnyxLocationPixelY { get; set; }
        public double OnyxLocationUmX { get; set; }
        public double OnyxLocationUmY { get; set; }
        public double OnyxUmHorizontalDisitance { get; set; }
        public double OnyxUmVerticalDisitance { get; set; }
        public double OnyxAngle { get; set; }

        public double ChassisLocationPixelX { get; set; }
        public double ChassisLocationPixelY { get; set; }

        public double ChassisLocationUmX { get; set; }
        public double ChassisLocationUmY { get; set; }

        public double ChassisUmHorizontalDisitance { get; set; }
        public double ChassisUmVerticalDisitance { get; set; }
        public double ChassisAngle { get; set; }

    }
}
