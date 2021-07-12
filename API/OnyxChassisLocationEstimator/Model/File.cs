using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Text;

namespace OnyxChassisLocationEstimator.Model
{
    public class File
    {

        public File()
        {

        }


        public string FileName { get; set; }

        public string FilePath { get; set; }


        public ObservableCollection<DrawObject> ResultCollection
        {
            get;set;
        }
    }
}
