from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FeatureNormalizer:
    def __init__(self, data, data_desc, features_to_normalize):
        self.frame = data
        self.desc = data_desc
        self.feat_norm = features_to_normalize
        self.normalized_frame = self.frame.copy()

    def normalize_features(self):
        for column in self.feat_norm:
            self.normalized_frame[column] = (self.frame[column] - self.desc[column]['mean']) /\
                                            (self.desc[column]['std'])

    def get_normalized_frame(self):
        self.normalize_features()
        return self.normalized_frame

