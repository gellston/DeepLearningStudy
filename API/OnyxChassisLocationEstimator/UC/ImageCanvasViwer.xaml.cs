using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace OnyxChassisLocationEstimator.UC
{
    /// <summary>
    /// ImageCanvasViewer.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class ImageCanvasViewer : UserControl, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        public void OnPropertyRaised(string propertyname)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(propertyname));
            }
        }

        public void Set<T>(string _name, ref T _reference, T _value)
        {
            if (!Equals(_reference, _value))
            {
                _reference = _value;
                OnPropertyRaised(_name);
            }
        }


        public ImageCanvasViewer()
        {
            InitializeComponent();

            //this.ZoomStep = 0.5;
            //this.ZoomMax = 5;
            //this.ZoomMin = 0.2;
            //this.Zoom = 1;


            CompositionTarget.Rendering += CompositionTarget_Rendering;


        }

        ~ImageCanvasViewer()
        {
            CompositionTarget.Rendering -= CompositionTarget_Rendering;

        }

        private void CompositionTarget_Rendering(object sender, EventArgs e)
        {
            this.InvalidateVisual();

        }

        public static readonly DependencyProperty ImageProperty = DependencyProperty.Register("Image", typeof(ImageSource), typeof(ImageCanvasViewer), new PropertyMetadata(OnCustomerChangedCallBack));
        public ImageSource Image
        {
            get
            {
                return (ImageSource)GetValue(ImageProperty);
            }

            set
            {
                SetValue(ImageProperty, value);
            }
        }

        private static void OnCustomerChangedCallBack(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ImageCanvasViewer control = sender as ImageCanvasViewer;
            if (control != null)
            {
                ImageSource image = e.NewValue as ImageSource;
                if (image == null) return;

                //if (control.ActualWidth == 0 || control.ActualHeight == 0) return;
                control.CanvasWidth = image.Width;
                control.CanvasHeight = image.Height;
                //control.Zoom = (image.Width > image.Height ? (control.ActualWidth / image.Width) : (control.ActualHeight / image.Height));
                //control.ZoomMax = control.Zoom * 20;
                //control.ZoomMin = control.Zoom / 20;
                //control.ZoomStep = (control.ZoomMax - control.ZoomMin) / 40;

                //control.OutScrollViewer.UpdateLayout();

                //control.OutScrollViewer.ScrollToVerticalOffset(control.OutScrollViewer.ScrollableHeight / 2);
                //control.OutScrollViewer.ScrollToHorizontalOffset(control.OutScrollViewer.ScrollableWidth / 2);

            }
        }

        private double _CanvasWidth = 0;
        public double CanvasWidth
        {
            get => _CanvasWidth;
            set => Set<double>(nameof(CanvasWidth), ref _CanvasWidth, value);
        }

        private double _CanvasHeight = 0;
        public double CanvasHeight
        {
            get => _CanvasHeight;
            set => Set<double>(nameof(CanvasHeight), ref _CanvasHeight, value);
        }

        public static readonly DependencyProperty ZoomProperty = DependencyProperty.Register("Zoom", typeof(double), typeof(ImageCanvasViewer), new PropertyMetadata(1.0));
        public double Zoom
        {
            get
            {
                return (double)GetValue(ZoomProperty);
            }

            set
            {
                SetValue(ZoomProperty, value);
            }
        }

        public static readonly DependencyProperty ZoomMaxProperty = DependencyProperty.Register("ZoomMax", typeof(double), typeof(ImageCanvasViewer), new PropertyMetadata(10.0));
        public double ZoomMax
        {
            get
            {
                return (double)GetValue(ZoomMaxProperty);
            }

            set
            {
                SetValue(ZoomMaxProperty, value);
            }
        }

        public static readonly DependencyProperty ZoomMinProperty = DependencyProperty.Register("ZoomMin", typeof(double), typeof(ImageCanvasViewer), new PropertyMetadata(0.1));
        public double ZoomMin
        {
            get
            {
                return (double)GetValue(ZoomMinProperty);
            }

            set
            {
                SetValue(ZoomMinProperty, value);
            }
        }

        public static readonly DependencyProperty ZoomStepProperty = DependencyProperty.Register("ZoomStep", typeof(double), typeof(ImageCanvasViewer), new PropertyMetadata(0.505));
        public double ZoomStep
        {
            get
            {
                return (double)GetValue(ZoomStepProperty);
            }

            set
            {
                SetValue(ZoomStepProperty, value);
            }
        }

        public static readonly DependencyProperty TranslationXProperty = DependencyProperty.Register("TranslationX", typeof(double), typeof(ImageCanvasViewer));
        public double TranslationX
        {
            get
            {
                return (double)GetValue(TranslationXProperty);
            }

            set
            {
                SetValue(TranslationXProperty, value);
            }
        }


        public static readonly DependencyProperty TranslationYProperty = DependencyProperty.Register("TranslationY", typeof(double), typeof(ImageCanvasViewer));
        public double TranslationY
        {
            get
            {
                return (double)GetValue(TranslationYProperty);
            }

            set
            {
                SetValue(TranslationYProperty, value);
            }
        }

        public static readonly DependencyProperty DrawObjectsCollectionProperty = DependencyProperty.Register("DrawObjectsCollection", typeof(ObservableCollection<Model.DrawObject>), typeof(ImageCanvasViewer));
        public ObservableCollection<Model.DrawObject> DrawObjectsCollection
        {
            get
            {
                return (ObservableCollection<Model.DrawObject>)GetValue(DrawObjectsCollectionProperty);
            }

            set
            {
                SetValue(DrawObjectsCollectionProperty, value);
            }
        }


        //public static readonly DependencyProperty SelectedItemProperty = DependencyProperty.Register("SelectedItem", typeof(HV.V1.Object), typeof(ImageCanvasViewer), new PropertyMetadata(OnSelectedItemChangedCallBack));
        //public HV.V1.Object SelectedItem
        //{
        //    get
        //    {
        //        return (HV.V1.Object)GetValue(SelectedItemProperty);
        //    }

        //    set
        //    {
        //        SetValue(SelectedItemProperty, value);
        //    }
        //}
        //private static void OnSelectedItemChangedCallBack(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        //{
        //    ImageCanvasViewer control = sender as ImageCanvasViewer;
        //    if (control != null)
        //    {

        //    }
        //}


        private Point CanvasStart;
        private Point CanvasOrigin;
        //private bool IsCanvasCaptured = false;
        private void OutScrollViewer_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            e.Handled = true;

            if (this.IsMouseCaptured == true)
                return;

            if (e.Delta > 0)
            {
                this.Zoom -= ((this.ZoomMax - this.ZoomMin) / 100);
                if (this.Zoom <= this.ZoomMin)
                    this.Zoom = this.ZoomMin;
            }
            else
            {
                this.Zoom += ((this.ZoomMax - this.ZoomMin) / 100);
                if (this.Zoom >= this.ZoomMax)
                    this.Zoom = this.ZoomMax;
            }

            OutScrollViewer.ScrollToVerticalOffset(OutScrollViewer.ScrollableHeight / 2);
            OutScrollViewer.ScrollToHorizontalOffset(OutScrollViewer.ScrollableWidth / 2);
        }

        private void ChildCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            e.Handled = true;
            Point cursorPosition = e.GetPosition(this);
            Canvas canvas = sender as Canvas;

            if (canvas != null)
            {
                if (canvas.IsMouseCaptured == true && canvas != null && Keyboard.IsKeyDown(Key.LeftShift))
                {
                    Vector v = this.CanvasStart - cursorPosition;
                    this.TranslationX = CanvasOrigin.X - v.X;
                    this.TranslationY = CanvasOrigin.Y - v.Y;
                    return;
                }

            }
        }

        private void ChildCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            Point pressedPoint = e.GetPosition(sender as IInputElement);

            HitTestResult hitTestResult = VisualTreeHelper.HitTest(sender as Visual, e.GetPosition(sender as IInputElement));
            if (hitTestResult == null) return;
            var element = hitTestResult.VisualHit;


            if (Keyboard.IsKeyDown(Key.LeftShift) == true && element.GetType() == typeof(Canvas))
            {
                this.CanvasStart = e.GetPosition(this);
                this.CanvasOrigin = new Point(this.TranslationX, this.TranslationY);

                var draggableControl = sender as Canvas;
                draggableControl.CaptureMouse();
                //IsCanvasCaptured = true;
                return;
            }

        }

        private void ChildCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            Canvas canvas = sender as Canvas;
            canvas.ReleaseMouseCapture();
            //this.SelectedItem = null;
            //this.IsCanvasCaptured = false;
        }

    }
}