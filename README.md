CNN - Command Words Recognition
=========
Speech recognition in Python using [Tensorflow 2](https://www.tensorflow.org/alpha) and [Keras](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras) high level API.  
Convolutional Neural Networks (__CNN__) were invented to classify time invariant data (like images). Recent researches found that, if sound is converted into its [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) (or better: [Log-Mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), then convolutional neural networks can be applied on sound's features (aka spectrogram) for training speech recognition model.
## Usage
1. Clone repository:  
    ```cli
    cd /path/to/these/files/
    git clone https://github.com/tyukesz/command-words-recognition-keras
    ```
2. Check __constants.py__ file for parameters like:  
    * DATASET_URL // URL where ZIPed sounds files are stored
    * SAMPLE_RATE // Recomended 16KHz
    * EPOCHS // number of training epochs
    * BATCH_SIZE //use power of 2: 64, 128, 256, 512, ...
    * TESTING and VALIDATION percentage // Recomended 15% - 15%
    * WANTED_WORDS // list of command words: ['yes', 'no', 'up', 'down', 'on', 'off', ...]
    * VERBOSITY //0=disable, 1=progressbar 
3. Install requirements:
    ```cli
    pip install requirements.txt
    ```
4. Run training on CPU:
    ```cli
    python train.py
    ```
    * If your are running training for first time it's recommended using '_--force_extract=True_' argument:
        ```cli
        python train.py --force_extract=True
        ```
    * If your __SOUNDS_DIR__ is empty it will download and extract the sounds files from provided __DATASET_URL__.
    * Forcing MFCC feature extraction (_force_extract=True_) causes saving sounds features in __MFCCS_DIR__ as tensors.
    * If features are already extracted, then the training begins faster since no need to recompute MFCCs of sounds, just load them as tensor.
5. \(Optional) You can load the pretrained model for [_transfer learning_](https://en.wikipedia.org/wiki/Transfer_learning):
    ```cli
    python train.py --load_model=name_of_your_model
    ```
6. \(Optional) Test your model prediction with WAV file:
    ```cli
    python predict_wav.py --load_model=name_of_your_model --wav_path=/path/to/yes.wav --num_of_predictions=2
    ```
    * The above command will output something like: 
        * yes (97%)
        * left (0.84%)  
        
## CNN architecure  
Using sequential Keras model with following layers:

 Layer (type)                 | Output Shape       | Params                          
 :---                         | :---               | :---                     
 conv2d (Conv2D)              | (512, 64, 32, 16)  | 416    
 max_pooling2d (MaxPooling2D) | (512, 32, 16, 16)  | 0   
 conv2d_1 (Conv2D)            | (512, 32, 16, 32)  | 4640
 max_pooling2d_1 (MaxPooling2)| (512, 16, 8, 32)   | 0       
 dropout (Dropout)            | (512, 16, 8, 32)   | 0       
| batch_normalization_v2 (BatchNormalization)      | (512, 16, 8, 32)   | 128    
 conv2d_2 (Conv2D)            | (512, 8, 4, 64)    | 18496   
 conv2d_3 (Conv2D)            | (512, 8, 4, 128)   | 73856   
 max_pooling2d_2 (MaxPooling2)| (512, 4, 2, 128)   | 0       
 dropout_1 (Dropout)          | (512, 4, 2, 128)   | 0       
 flatten (Flatten)            | (512, 1024)        | 0       
 dropout_2 (Dropout)          | (512, 1024)        | 0       
 dense (Dense)                | (512, 256)         | 262400  
 dense_1 (Dense)              | (512, 6)           | 1542    
| ____________________________|
| Total params: 361,478| 
| Trainable params: 361,414| 
| Non-trainable params: 64| 

## License
This project is licensed under the MIT License  - see the [LICENSE.md](LICENSE.md) file for details.   
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)