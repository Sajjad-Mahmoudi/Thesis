using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;
using System.Diagnostics;
using System.Collections.Generic;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace BinarySeg
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            InitializeComponent();
        }

        [ComImport]
        [Guid("5b0d3235-4dba-4d44-865e-8f1d0e4fd04d")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private unsafe interface IMemoryBufferByteAccess
        {
            void GetBuffer(out byte* buffer, out uint capacity);
        }

        private modelModel modelGen;
        private readonly modelInput modelInput = new modelInput();
        private modelOutput modelOutput;

        //Load a machine learning model
        private async Task<bool> LoadModelAsync()
        {
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/model.onnx"));
            Task<modelModel> loadTask = modelModel.CreateFromStreamAsync(modelFile);
            try
            {
                modelGen = await loadTask;
                return true;
            }
            catch(System.Exception e)
            {
                return false;
            }
        }

        // Link to a button

        bool isBusy = false;
        public async void UpdateFromFolderCallback(object send, RoutedEventArgs e)
        {
            if (isBusy)
                return;

            isBusy = true;
            bool result = await LoadModelAsync();

            if (!result)
                return;

            TimeSpan accumulatedTimeSpan = new TimeSpan();

            // Prompt folder
            StorageFolder srcfolder = await OpenFolder();
            StorageFolder gtfolder = await OpenFolder();
            if (srcfolder == null || gtfolder == null)
            {
                return;
            }

            // Get all files in folder
            var srcfiles = await srcfolder.GetFilesAsync();
            var gtfiles = await gtfolder.GetFilesAsync();
            if (srcfiles.Count != gtfiles.Count)
            {
                return;
            }

            List<double> times = new List<double>();
            
            for (int i = 0; i < srcfiles.Count; i++)
            {
                var src = srcfiles[i];
                var gt = gtfiles[i];

                // Get images
                var srcimage = await ReadImage(src);
                var gtimage = await ReadImage(gt);

                if (srcimage == null || gtimage == null)
                {
                    continue;
                }

                // Preprocess, prediction, inference time calculation, and post process
                var imageInput = PreprocessImage(srcimage);
                var sw = Stopwatch.StartNew();
                var imagePred = await InferenceAsync(imageInput);
                var s = sw.Elapsed;
                accumulatedTimeSpan += s;
                var imageResult = await PostProcess(imagePred);

                var dt = s.TotalSeconds;
                times.Add(dt);

                // Interface (display images and inference times)
                await UpdateImageAsync(sourceImageWindow, srcimage);
                await UpdateImageAsync(labelImageWindow, gtimage);
                await UpdateImageAsync(resultImageWindow, imageResult);
                TimeSpanTextBox.Text = $"Inference Time (ms) = {Math.Round(dt*1000, 0):0}";
                //TimeSpanTextBox.Text = $"Inference Time (ms) = {s.TotalMilliseconds:F2}";
                TotalTimeSpanTextBox.Text = $"Total Inference Time (s) = {accumulatedTimeSpan.TotalSeconds:0.00}";

                await Task.Delay(100);
            }


            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            foreach (var t in times)
                sb.AppendLine(t.ToString());

            var pics = Windows.Storage.KnownFolders.PicturesLibrary;
            //var file = await pics.CreateFileAsync("results.txt");
            //await Windows.Storage.FileIO.WriteTextAsync(file, sb.ToString());

            TimeSpanTextBox.Text = "DONE!";

            isBusy = false;
        }
        
        // Convert Softwarebitmap to TensorFloat
        private unsafe TensorFloat PreprocessImage(SoftwareBitmap image)
        {
            // In BGRA8 format, each pixel is defined by 4 bytes
            const int BYTES_PER_PIXEL = 4;
            float[,,,] floatArray = new float[1, 256, 256, 1];

            using (var buffer = image.LockBuffer(BitmapBufferAccessMode.ReadWrite))
            using (var reference = buffer.CreateReference())
            {
                unsafe
                {
                    // Get a pointer to the pixel buffer
                    ((IMemoryBufferByteAccess)reference).GetBuffer(out byte* data, out uint capacity);

                    // Get information about the BitmapBuffer
                    var desc = buffer.GetPlaneDescription(0);

                    // Iterate over all pixels
                    for (uint row = 0; row < desc.Height; row++)
                    {
                        for (uint col = 0; col < desc.Width; col++)
                        {
                            // Index of the current pixel in the buffer (defined by the next 4 bytes, BGRA8)
                            var currPixel = desc.StartIndex + desc.Stride * row + BYTES_PER_PIXEL * col;

                            // Read the current pixel information into b,g,r channels (leave out alpha channel)
                            var b = data[currPixel + 0]; // Blue
                            var g = data[currPixel + 1]; // Green
                            var r = data[currPixel + 2]; // Red

                            // populate float array
                            floatArray[0, row, col, 0] = b / 255.0f;
                        }
                    }

                }
            }
            // create TensorFloat to input the ONNX model
            TensorFloat input = TensorFloat.CreateFromArray(new long[] { 1, 256, 256, 1 }, floatArray.Cast<float>().ToArray());
            return input;
        }
        
        // Bind input, evaluate model, and bind output to the model
        private async Task<modelOutput> InferenceAsync(TensorFloat image)
        {
            //Bind the preprocessed input to the model
            modelInput.input = image;

            // Evaluate the model
            modelOutput = await modelGen.EvaluateAsync(modelInput);
            return modelOutput;
        }

        private async Task<SoftwareBitmap> PostProcess(modelOutput modelOutput)
        {
            var pixels = modelOutput.conv2d_18
                        .GetAsVectorView()
                        .SelectMany(
                                    f =>
                                    {
                                        f = f < 0.5 ? 0 : 1;  //if f<0.5 then f=0 else f=1
                                        byte v = Convert.ToByte(f * 255);
                                        return new byte[] { v, v, v, v};
                                    })
                        .ToArray();

            var writeableBitmap = new WriteableBitmap(256, 256);

            // Open a stream to copy the image contents to the WriteableBitmap's pixel buffer 
            using (Stream stream = writeableBitmap.PixelBuffer.AsStream())
            {
                await stream.WriteAsync(pixels, 0, pixels.Length);
            }

            SoftwareBitmap outputBitmap = SoftwareBitmap.CreateCopyFromBuffer(writeableBitmap.PixelBuffer, BitmapPixelFormat.Bgra8, 256, 256);
            return outputBitmap;
        }

        // Opens a folder
        private async Task<StorageFolder> OpenFolder()
        {
            var folderPicker = new Windows.Storage.Pickers.FolderPicker
            {
                SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary,
            };
            folderPicker.FileTypeFilter.Add("*");

            StorageFolder folder = await folderPicker.PickSingleFolderAsync();
            if (folder != null)
            {
                // Application now has read/write access to all contents in the picked folder
                // (including other sub-folder contents)
                Windows.Storage.AccessCache.StorageApplicationPermissions.
                FutureAccessList.AddOrReplace("PickedFolderToken", folder);
                return folder;
            }
            else
            {
                return null;
            }
        }

        // Opens as StorageFile as an image --> SoftwareBitmap
        private async Task<SoftwareBitmap> ReadImage(StorageFile inputFile)
        {
            // Init empty image
            SoftwareBitmap bitmap;

            using (IRandomAccessStream stream = await inputFile.OpenAsync(FileAccessMode.Read))
            {
                switch (inputFile.FileType.ToLower())
                {
                    case ".png":
                        // Create the decoder from the stream
                        var decoder = await BitmapDecoder.CreateAsync(stream);
                        bitmap = await decoder.GetSoftwareBitmapAsync();
                        break;

                    default:
                        bitmap = null;
                        break;
                }
            }

            // Check if image is empty
            if (bitmap == null)
            {
                return null;
            }

            if (bitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8)
            {
                bitmap = SoftwareBitmap.Convert(bitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore);
            }

            return bitmap;
        }

        // Set the UI image window to a new image
        private async Task UpdateImageAsync(Image imageWindow, SoftwareBitmap bitmap)
        {
            // Safety
            if (imageWindow == null || bitmap == null)
            {
                return;
            }

            // Check format
            if (bitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8 || bitmap.BitmapAlphaMode != BitmapAlphaMode.Ignore)
            {
                bitmap = SoftwareBitmap.Convert(bitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore);
            }

            // Set image to source
            var imageSource = new SoftwareBitmapSource();
            await imageSource.SetBitmapAsync(bitmap);

            // Set image window
            imageWindow.Source = imageSource;
        }
    }
}
