using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace OnyxChassisLocationEstimator.Model
{
    public class DrawObject : INotifyPropertyChanged
    {
        public DrawObject()
        {

        }

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

        private int _Width = 0;
        public int Width
        {
            get => _Width;
            set => Set(nameof(Width),ref _Width, value);
        }

        private int _Height = 0;
        public int Height
        {
            get => _Height;
            set => Set(nameof(Height), ref _Height, value);
        }

        private int _X = 0;
        public int X
        {
            get => _X;
            set => Set(nameof(X), ref _X, value);
        }

        private int _Y = 0;
        public int Y
        {
            get => _Y;
            set => Set(nameof(Y), ref _Y, value);
        }
    }

    
}
