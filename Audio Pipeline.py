import librosa as lb
import numpy as np
import os
import pickle as pkl
#loads an audio file
class Loader:
    def __init__(self, sampleRate, mono, duration):
        self.sampleRate= sampleRate
        self.mono = mono
        self.duration = duration

    def load(self, filePath):
        audioSignal = lb.load(filePath,
                              sr=self.sampleRate,
                              duration= self.duration,
                              mono=self.mono)[0]
        return audioSignal

#applies padding to an array if necessary
class Pader:
    def __init__(self, mode="constant"):
        self.mode = mode

    #[1,2,3] -> [0,0,1,2,3] (makes all arrays the same size)
    def __leftPad__(self, array, numMissingItems):
        paddedArray = np.pad(array,
                             (numMissingItems,0),
                             mode = self.mode)
        return paddedArray

    #[1,2,3] -> [1,2,3,0,0]
    def __rightPad__(self, array, numMissingItems):
        paddedArray = np.pad(array,
                             (0, numMissingItems),
                             mode= self.mode)
        return paddedArray

#extracts a mel spectrogram from an audio signal
class MelSpectrogram:
    def __init__(self, frameSize, hopLength):
        self.frameSize = frameSize
        self.hopLength = hopLength

    def extract(self,signal):
        stft = lb.stft(signal,
                       n_fft=self.frameSize,
                       hop_length=self.hopLength)[:-1]
        spectrogram = np.abs(stft)
        logSpectrogram = lb.amplitude_to_db(spectrogram)
        return logSpectrogram

#applies min max normalisation to an array
class MinMaxNorm:
    def __init__(self, minVal, maxVal):
        self.min = minVal
        self.max = maxVal

    def normalise(self, array):
        #puts the array values from 0 to 1
        normArray = (array - array.min())/ (array.max() - array.min())
        #squishes all values between the min and max value
        normArray = normArray * (self.max -self.min) + self.min
        return normArray

    def __denormalize__(self, normArray, ogMin, ogMax):
        array = (normArray - self.min) / (self.max-self.min)
        array = array *(ogMax - ogMin) + ogMin
        return array

    #saves features and the min max values of the spectrogram
class save:
    def __init__(self, featureSaveDir, minMaxValDir):
        self.featureSaveDir = featureSaveDir
        self.minMaxValDir = minMaxValDir

    def saveFeature(self, feature, filePath):

        np.save(self.generateSavePath(filePath), feature)

    def saveMinMaxValues(self, minMaxVals):
        savePath = os.path.join(self.minMaxValDir,
                                "minMaxValues.pkl")
        self.save(minMaxVals, savePath)


    @staticmethod
    def save(data, savePath):
        with open(savePath, "wb") as f:
            pkl.dump(data, f)


    def generateSavePath(self,filePath):
        fileName = os.path.split(filePath)[1]
        savePath = os.path.join(self.featureSaveDir + fileName + ".npy")
        return savePath



# processes audio in a file directory, doing the following steps to each file
# 1. load the file
# 2. pad the signal
# 3. extract spectrogram from signal
# 4. normalise spectrogram
# 5. save spectrogram to a file
# 6. save min max value for all spectrograms

class Preprocessing:

    def __init__(self):
        self._loader = None
        self.pader = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.minMaxVals = {}
        self.numExpectedSamples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self.numExpectedSamples = int(loader.sampleRate * loader.duration)

    def process(self, audioFileDir):
        #loops through the audio files in the directory
        for root, _, files in os.walk(audioFileDir):
            for file in files:
                filePath = os.path.join(root, file)
                self.processFile(filePath)
                print(f"processed file {filePath}")
            self.saver.saveMinMaxValues(self.minMaxVals)

    def processFile(self, filePath):
        signal = self.loader.load(filePath)
        if self.isPaddingRequired(signal):
            signal = self.applyPadding(signal)
        feature = self.extractor.extract(signal)
        normalFeature = self.normaliser.normalise(feature)
        savePath = self.saver.saveFeature(normalFeature, filePath)
        self.storeMinMaxVal(savePath, feature.min(), feature.max())

    def isPaddingRequired(self,signal):
        numExpectedSamples = int(self.loader.sampleRate * self.loader.duration)

        if len(signal) < numExpectedSamples:
            return True
        return False

    def applyPadding(self, signal):
        numMissingSamples = self.numExpectedSamples - len(signal)
        paddedSignal = self.pader.rightPad(signal, numMissingSamples)
        return paddedSignal

    def storeMinMaxVal (self, savePath, minVal, maxVal):
        self.minMaxVals[savePath] = {
            "min": minVal,
            "max": maxVal
        }

if __name__ == "__main__":
    frameSize = 512
    hopLength = 256
    duration = .75
    sampleRate = 22050
    mono = True

    spectrogramSaveDir = "/Users/antoinenguyen/Downloads/data/Parsed_Pump_noise_spectrogram/"
    minMaxValueSaveDir = "/Users/antoinenguyen/Downloads/data/MinMaxValue/"
    fileDir = "/Users/antoinenguyen/Downloads/data/Data_2/"

    #instantiate all objects

    loader = Loader(sampleRate, duration, mono)
    pader = Pader()
    logSpectrogramExtractor = MelSpectrogram(frameSize,hopLength)
    minMaxNorm = MinMaxNorm(0,1)
    saver = save(spectrogramSaveDir,minMaxValueSaveDir)

    preprocessing = Preprocessing()
    preprocessing.loader = loader
    preprocessing.pader = pader
    preprocessing.extractor = logSpectrogramExtractor
    preprocessing.normaliser = minMaxNorm
    preprocessing.saver = saver

    preprocessing.process(fileDir)

