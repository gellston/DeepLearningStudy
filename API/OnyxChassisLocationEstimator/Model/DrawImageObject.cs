using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Media.Imaging;

namespace OnyxChassisLocationEstimator.Model
{
    public class DrawImageObject : DrawObject
    {
        public DrawImageObject()
        {

        }


        private WriteableBitmap _Image = null;
        public WriteableBitmap Image
        {
            get => _Image;
            set => Set(nameof(Image), ref _Image, value);
        }
    }
}
