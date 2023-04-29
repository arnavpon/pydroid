import numpy as np
import pandas as pd
from scipy.signal import resample

from pipeline_v1.chrominance import chrominance, CHROM_SETTINGS
from pipeline_v1.ieee_video_start_frames import StartFrames
from pipeline_v1.signal_pross import (
    normalize_signal,
    detrend_w_poly,
    bandpass,
    min_max_scale,
    perform_ica
)
from pipeline_v1.wavelet import apply_wavelet

RGB_RATE = 30  # video frame rate
BVP_RATE = 64  # ground truth BVP sample rate


class IeeeGroundTruth:
    """
    Class for loading and processing RGB data collected from the IEEE dataset. Most importantly,
    it lines up the RGB data with the corresponding BVP data.
    """

    def __init__(self, subject, trial = 1, directory = 'channel_data3'):
        
        # IEEE dataset organizes the subjects into subjects and trials
        # we only use the first trial for each subject
        self.subject = subject
        self.trial = trial
        
        self.directory = directory
        self.rgb_freq = RGB_RATE
        self.bvp_freq = BVP_RATE

        self.rgb = None
        self.bvp = None
    
    def align_rgb_bvp(self):
        """
        Align RGB data with its corresponding ground truth BVP.
        """

        # get rgb, and ground truth bvp and ibi data from files
        rgb = self.get_rgb_data()
        bvp = self.get_bvp_data()
        self.ibis = self.get_ibi_data()

        # get the event markers
        event_markers = self.get_event_marker_data()

        # get start and end point for alignment
        # taken from: https://github.com/mxahan/project_rppg/blob/master/Codes/Data_read_only/data_read_main.py
        start_gap =  event_markers[0] - bvp[0] 
        end_point =  event_markers[1] - event_markers[0]

        # hard coded start frames from each tags.csv file for the IEEE data
        start_frame = StartFrames[self.subject][self.trial]

        # also taken from https://github.com/mxahan/project_rppg/blob/master/Codes/Data_read_only/data_read_main.py
        self.rgb = rgb[start_frame: start_frame + int(end_point * 30) + 5]
        self.bvp = bvp[int(start_gap * 64): int((start_gap + end_point) * 64)]
    
    def align_indices(self, rgb_index):
        """
        Given the index of a frame in the RGB data, return the corresponding index of ground truth
        BVP data.
        """
        return int(rgb_index * (self.bvp_freq / self.rgb_freq))

    def fill_nans(self):
        """
        Fill nans in the data through interpolation or setting values to equal their
        neighbors, if it's the first value.
        """

        if self.rgb is None or self.bvp is None:
            raise Exception('Must call align_rgb_bvp before calling fill_nans.')

        # make sure first element of each isn't nan
        for i in range(self.rgb.shape[1]):
            if np.isnan(self.rgb[0, i]):
                
                # get the first non-nan value
                for val in self.rgb[1: , i]:
                    if not np.isnan(val):
                        self.rgb[0, i] = val
                        break
        
        # same for bvp
        if np.isnan(self.bvp[0]):
            for val in self.bvp[1: ]:
                if not np.isnan(val):
                    self.bvp[0] = val
                    break
        
        # use pandas to linearly interpolate the rest of the nans
        for i in range(self.rgb.shape[1]):
            self.rgb[:, i] = pd.Series(self.rgb[:, i]).interpolate(method = 'linear').to_numpy()
        self.bvp = pd.Series(self.bvp.flatten()).interpolate(method = 'linear').to_numpy()
        
    def process_rgb(self, minmax = False, use_wavelet = True, use_bandpass = False):
        self.rgb = self.process_rgb_general(
            self.rgb,
            minmax = minmax,
            use_wavelet = use_wavelet,
            use_bandpass = use_bandpass,
            rgb_freq = self.rgb_freq
        )

    @staticmethod
    def process_rgb_general(given_rgb, minmax = False, use_wavelet = True, use_bandpass = False, rgb_freq = RGB_RATE):
        """
        Preprocessing for RGB data. Turning minmax on or off toggles whether we use minmax scaling or if
        we normalize by subtracting the mean and dividing by the standard deviation. Turning use_wavelet on
        filters the data using wavelet filter. Turning use_bandpass on filters the data using a bandpass filter.
        I use minmax = False, use_wavelet = True, and use_bandpass = False in the thesis.
        """

        rgb = given_rgb.copy()

        # detrend each channel
        for i in range(rgb.shape[1]):
            rgb[:, i] = detrend_w_poly(rgb[:, i])
        
        if use_wavelet:
            for i in range(rgb.shape[1]):
                rgb[:, i] = apply_wavelet(
                    rgb[:, i],

                    # playing with these params could yield different or maybe better results,
                    # I didn't get a chance to do more with them 
                    wave = 'db2',
                    level = 2
                )
        
        if use_bandpass:
            for i in range(rgb.shape[1]):
                rgb[:, i] = bandpass(
                    rgb[:, i],
                    rgb_freq,
                    [0.5, 3],  # altering this could yield better results
                    order = 4
                )

        if minmax:
            for i in range(rgb.shape[1]):
                rgb[:, i] = min_max_scale(rgb[:, i])
        else:
            for i in range(rgb.shape[1]):
                rgb[:, i] = normalize_signal(rgb[:, i])
        
        return rgb

    
    def process_bvp(self):
        """
        Preprocess ground truth BVP by normalizing and detrending.
        """
        self.bvp = normalize_signal(self.bvp)
        self.bvp = detrend_w_poly(self.bvp)
    
    def prepare_data_for_ml(self, num_feats_per_channel = 5, skip_amount = 10):
        """
        Bring together RGB and BVP into a single dataframe for easier handling at the ML step. RGB data
        is upsampled to the frequency of the BVP data to that the model output can be directly compared
        to the ground truth BVP.

        Also performs feature engineering step. Features explained below:
        - velocity features: equivalent of the 1st derivative of the RGB channels
        - acceleration features: equivalent of the 2nd derivative of the RGB channels
        - memory features: previous state of each channel; parametric in now many there are and how far back they go
        - chrom: based on the Chrominance algorithm
        - ICA: performs independent component analysis on the RGB channels, selects the returned component with the
            highest spectral power, and uses that as a feature
        """

        # upsample RGB
        rgb_upsampled = np.zeros((len(self.bvp), self.rgb.shape[1]))
        for col in range(rgb_upsampled.shape[1]):
            rgb_upsampled[:, col] = self.upsample_signal(
                self.rgb[:, col],
                len(self.bvp),
            )
        
        df = self.prepare_data_for_ml_general(
            rgb_upsampled,
            num_feats_per_channel,
            skip_amount,
            upsample = False,
            rgb_freq = self.rgb_freq,
            bvp_freq = self.bvp_freq
        )

        bvp_in_use = self.bvp[2: ]  # discard first 2 elements to align with vel and acc feats
        bvp_in_use = bvp_in_use[num_feats_per_channel * skip_amount: ]  # to align with memory feats now
        df['bvp'] = bvp_in_use
        return df

    @staticmethod
    def prepare_data_for_ml_general(rgb, num_feats_per_channel = 5, skip_amount = 10, upsample = True,
                                    rgb_freq = RGB_RATE, bvp_freq = BVP_RATE):

        if upsample:

            # get new length for the rgb
            new_len = int((bvp_freq / rgb_freq) * len(rgb))
            
            # init new array to hold upsampled rgb
            rgb_upsampled = np.zeros((new_len, rgb.shape[1]))
            
            # upsample each channel individually
            for col in range(rgb_upsampled.shape[1]):
                rgb_upsampled[:, col] = resample(rgb[:, col], new_len)
        
        else:
            rgb_upsampled = rgb.copy()

        # get "velocity" and "acceleration" features of rgb
        rgb_vel = np.zeros((rgb_upsampled.shape[0] - 2, rgb_upsampled.shape[1]))
        rgb_acc = np.zeros((rgb_upsampled.shape[0] - 2, rgb_upsampled.shape[1]))
        for col in range(rgb_vel.shape[1]):
            vel = np.diff(rgb_upsampled[:, col], n = 1)[1: ]  # need to drop 1st element to align w/ accel feats
            acc = np.diff(rgb_upsampled[:, col], n = 2)
            rgb_vel[:, col] = vel
            rgb_acc[:, col] = acc

        rgb_upsampled = rgb_upsampled[2: , :]  # discard first 2 elements to align with vel and acc feats
        
        # get chrom and ICA feats
        chrom = IeeeGroundTruth.prepare_chrominance_as_feature(rgb_upsampled)
        ica_feat = perform_ica(rgb_upsampled)

        # get memory feats
        mems = IeeeGroundTruth.get_memory_features(rgb_upsampled, num_feats_per_channel, skip_amount)
        
        # adjust length of all features to align with memory feats
        rgb_upsampled = rgb_upsampled[num_feats_per_channel * skip_amount: , :]
        chrom = chrom[num_feats_per_channel * skip_amount: ]
        rgb_vel = rgb_vel[num_feats_per_channel * skip_amount: , :]
        rgb_acc = rgb_acc[num_feats_per_channel * skip_amount: , :]
        ica_feat = ica_feat[num_feats_per_channel * skip_amount: ]

        # collect the data (except memory) in a map
        data = {
            'r': rgb_upsampled[:, 0],
            'g': rgb_upsampled[:, 1],
            'b': rgb_upsampled[:, 2],
            'r_vel': rgb_vel[:, 0],
            'g_vel': rgb_vel[:, 1],
            'b_vel': rgb_vel[:, 2],
            'r_acc': rgb_acc[:, 0],
            'g_acc': rgb_acc[:, 1],
            'b_acc': rgb_acc[:, 2],
            'chrom': chrom,
            'ica_feat': ica_feat,
        }

        for k, v in mems.items():
            data[k] = v
        
        return pd.DataFrame(data)

    @staticmethod
    def get_memory_features(rgb_upsampled, num_feats_per_channel, skip_amount):
        """
        Get the memory features, as described above.
        """

        # create a dictionary with a key for each memory feature
        channels = ['r', 'g', 'b']
        mems = {f'{c}_mem_{i}': [] for i in range(num_feats_per_channel) for c in channels}
        
        # populate all the memory features
        for idx in range(num_feats_per_channel * skip_amount, rgb_upsampled.shape[0]):
            for i, c in enumerate(channels):
                for j in range(num_feats_per_channel):
                    mems[f'{c}_mem_{j}'].append(rgb_upsampled[idx - (j * skip_amount), i])
        
        return mems

    def get_rgb_data(self):
        return pd.read_csv(
            f'{self.directory}/ieee-subject-{self.subject:03d}-trial-{self.trial:03d}.csv'
        ).to_numpy()

    def get_bvp_data(self):
        return pd.read_csv(
            f'pipeline_v1/validation_data/IEEE_data/subject_{self.subject:03d}/trial_{self.trial:03d}/empatica_e4/BVP.csv',
            header = None
        ).to_numpy()

    def get_ibi_data(self):
        return pd.read_csv(
            f'pipeline_v1/validation_data/IEEE_data/subject_{self.subject:03d}/trial_{self.trial:03d}/empatica_e4/IBI.csv',
            header = None
        ).to_numpy()[1: , 1].flatten()

    def get_event_marker_data(self):
        return pd.read_csv(
            f'pipeline_v1/validation_data/IEEE_data/subject_{self.subject:03d}/trial_{self.trial:03d}/empatica_e4/tags.csv'
        ).to_numpy()

    @staticmethod
    def upsample_signal(signal, new_length):
        return resample(signal, new_length)

    @staticmethod
    def prepare_chrominance_as_feature(rgb):
        """
        Run the chrominance algorithm on the RGB data and then apply some postprocessing after
        the fact. Note that the postprocessing shouldn't really be necessary, but the model's
        performance seemed better with the postprocessng for some reason.
        """

        chrom = chrominance(rgb, CHROM_SETTINGS)

        # postprocessing
        chrom = apply_wavelet(chrom, 'db2', 2)
        chrom = bandpass(chrom, 64, [0.67, 3.0], 4)
        chrom = min_max_scale(chrom)
        
        return chrom
