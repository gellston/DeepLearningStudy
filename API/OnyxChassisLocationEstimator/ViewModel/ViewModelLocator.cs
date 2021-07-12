using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.Ioc;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace OnyxChassisLocationEstimator.ViewModel
{
    public class ViewModelLocator
    {

        public ViewModelLocator()
        {

            SimpleIoc.Default.Register<MainWindowViewModel>();

            var currentDirectory = System.AppDomain.CurrentDomain.BaseDirectory;
            var resultFolder = currentDirectory + "Result";
            Directory.CreateDirectory(resultFolder);
            

        }



        public ViewModelBase MainWindowViewModel
        {
            get => SimpleIoc.Default.GetInstance<MainWindowViewModel>();
        }




    }
}
