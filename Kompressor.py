import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
import scipy.signal as sig
import scipy.fft as fft







class Kompressor:
    def __init__(self,
                 threshold_compressor:float,
                 ratio_compressor:float,
                 threshold_expander:float,
                 ratio_expander:float,
                 filename:str):
        self.threshold_compressor = threshold_compressor
        self.threshold_expander = threshold_expander
        self.ratio_compressor = ratio_compressor
        self.ratio_expander = ratio_expander
        
        self.data = self.load_data(filename)
        
    def load_data(self,path:str)->np.ndarray:
        data, fs = librosa.load(path)
        self.fs = fs
        return data
        
    def apply_compressor(self,data,threshold,ratio):
        rms = np.sqrt(data**2)
        dbfs = 20 * np.log10(rms.mean())
        gain_factor = 10 ** ((dbfs+ratio*dbfs - dbfs) / 20)
        #verstärkung
        gain = np.where(np.abs(rms) > threshold,gain_factor, 1.0)
        output_signal = data * gain
        
        
        #muss threshold vlt als Pegel angegeben werden und dann immer alles mit Pegeln gemahcr werden
        
        return output_signal
    
    def apply_expander(self,data,threshold,ratio):
        rms = np.sqrt(data**2)
        gain = np.where(np.abs(rms) < threshold, ratio, 1.0)

        # Anwenden der Verstärkung
        output_signal = data * gain
        return output_signal
    
    
    
        
    def block_processing(self,data): #das vlt doch nicht als methode machen sondern spöter unten als programm durchlauf
        #do="filter" und argumente übergeben das dann mit getatr auf das zugegriffen werden kann
        #create filter
        h = sig.firwin(1,0.1,fs=fs)
        M = len(h)
        Nx = len(data)
        #next pow to
        N = int(8 * 2**np.ceil(np.log2(M)))
        #stepsize
        step_size = N - (M - 1)
        H = fft.fft(h, N)
        #create zero vector for end signal
        y = np.zeros(Nx + M - 1, dtype=np.complex128)
        position = 0
        #apply block porcessing
        while position + step_size <= Nx:
            #get current block and make fft
            current_block = data[position : position + step_size]
            X_block = fft.fft(current_block, N)
            #apply window
            Y_block = fft.ifft(X_block * H)
            # self.filterbank(Y_block)
            #hier soll dann mit getatr was ausgeführt werden
            #add data on zero vector on the right position
            y[position : position + N] += Y_block
            position += step_size
        return y.real 
    
    def filterbank(self,data,gain=[10,-30,-100]):
        #hier noch herausfinden was keine Verstärkung oder Dämpfung hervorrufen würde
        #lowpass filter
        b_low,a_low = sig.butter(4,1000,"lowpass",fs=self.fs)
        low_signal = sig.lfilter(b_low,a_low,data)
        low_signal = self.apply_gain(low_signal,gain[0]) #hier mal schaeu wie ich noch rechenleistung soaren kann
        #bandpass filter
        b_mid,a_mid = sig.butter(4,[1000,4000],"bandpass",fs=self.fs)
        mid_signal = sig.lfilter(b_mid,a_mid,data)
        mid_signal = self.apply_gain(mid_signal,gain[1])
        #highbanf filter
        b_high,a_high = sig.butter(4,4000,"highpass",fs=self.fs)
        high_signal = sig.lfilter(b_high,a_high,data)
        high_signal = self.apply_gain(high_signal,gain[2])
        #add the bands together again
        signal = low_signal + mid_signal + high_signal
        return signal
        
    
    def calc_pegel(self,data)->tuple[int,np.ndarray]: #das hier in dbfs umbasteln
        rms = np.sqrt(data**2)
        signal = 10*np.log10(rms/1) 
        pegel = signal.mean()
        # print(f"Der Pegel beträgt: {pegel:.2f}")
        return pegel
        
        
    #liiter als property implementieren        
    def apply_limiter(self,data:np.ndarray)->np.ndarray:
        #kann der auch irgendwie als property arbeiten?
        
        rms = np.sqrt(data**2)#calc rms
        #limiter: das vlt doch nicht richtgig
        rms = np.where(rms>threshold,rms,threshold)
        return rms 
    

    
    def apply_gain(self,data,pegel):
        rms = np.sqrt(np.mean(np.square(data)))
        max_amplitude = 1.0
        dbfs = 20 * np.log10(rms / max_amplitude)
        gain_factor = 10 ** ((pegel - dbfs) / 20)
        amplified_signal = data * gain_factor
        return amplified_signal
    
    def makeup_gain(self,data,threshold,ratio):
        gain = (1-1/ratio)
        data = np.where(np.abs(data)>threshold,data,data*gain)
        
        #calc pegel apply gain und dann wider zurück rechnen
        
        return data











if __name__ == "__main__":
    path = r"bass_T-20_R2_Mg6_sK5_Ar1.00047_Rr0.99969.wav"
    #read data
    data, fs = librosa.load(path)
    speaker = sc.default_speaker()
    threshold_kompressor = 0.9
    threshold_expander = 0.1
    ratio_kompressor = 0.2
    ratio_expander = 3


    kompressor = Kompressor(threshold_kompressor,threshold_expander,0.5)
    #vlt wenn das beim klassenaufruf argumente nimmt, dann nicht nochmal in den Methoden
    #alles bei klassenaufruf definieren, und auch sachen berechen wie zum Beispiel den RMS

    #trotzdem mal hier kompressor und expander hinbekommen!

    speaker.play(data[:3*fs],fs)


    plt.plot(data)

    plt.show()






    #das ganze soll vlt etwas dynamischer werden, das in pegel berechnet wird und dann outpegel definiert ist und die Verstärkung 
    #anhand dessen berechnet wird. ->glaube so könnte das funktionieren mit der kompressor kennlinie.
    #mehr die Pegel integrieren?