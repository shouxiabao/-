import torch
import esm
import numpy as np


class AMPFeatureExtractor:
    """抗菌肽特征提取器（基于ESM2）"""

    def __init__(self):
        """初始化特征提取器"""
        print("正在加载ESM2模型...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载ESM模型
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()
        print("✅ ESM2模型加载完成!")

    def extract_sequence_features(self, sequences):
        """提取序列特征"""
        if not sequences:
            return np.array([])

        batch_data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[6], return_contacts=True)

        embeddings = results["representations"][6].cpu()
        features = []

        for i, seq in enumerate(sequences):
            seq_length = len(seq)
            seq_embeddings = embeddings[i, 1:seq_length + 1]

            # 关键特征
            mean_pooled = seq_embeddings.mean(dim=0)
            max_pooled = seq_embeddings.max(dim=0)[0]
            cls_feature = embeddings[i, 0]

            # 物理化学特征
            physchem_features = self._extract_physicochemical_features(seq)

            # 组成特征
            composition_features = self._extract_composition_features(seq)

            combined_features = torch.cat([
                mean_pooled,
                max_pooled,
                cls_feature,
                torch.tensor(physchem_features).float(),
                torch.tensor(composition_features).float(),
                torch.tensor([seq_length]).float()
            ])

            features.append(combined_features.numpy())

        return np.array(features)

    def _extract_physicochemical_features(self, sequence):
        """提取物理化学特征"""
        aa_properties = {
            'A': {'hydropathy': 1.8, 'charge': 0, 'polarity': 0},
            'R': {'hydropathy': -4.5, 'charge': 1, 'polarity': 1},
            'N': {'hydropathy': -3.5, 'charge': 0, 'polarity': 1},
            'D': {'hydropathy': -3.5, 'charge': -1, 'polarity': 1},
            'C': {'hydropathy': 2.5, 'charge': 0, 'polarity': 0},
            'Q': {'hydropathy': -3.5, 'charge': 0, 'polarity': 1},
            'E': {'hydropathy': -3.5, 'charge': -1, 'polarity': 1},
            'G': {'hydropathy': -0.4, 'charge': 0, 'polarity': 0},
            'H': {'hydropathy': -3.2, 'charge': 0.5, 'polarity': 1},
            'I': {'hydropathy': 4.5, 'charge': 0, 'polarity': 0},
            'L': {'hydropathy': 3.8, 'charge': 0, 'polarity': 0},
            'K': {'hydropathy': -3.9, 'charge': 1, 'polarity': 1},
            'M': {'hydropathy': 1.9, 'charge': 0, 'polarity': 0},
            'F': {'hydropathy': 2.8, 'charge': 0, 'polarity': 0},
            'P': {'hydropathy': -1.6, 'charge': 0, 'polarity': 0},
            'S': {'hydropathy': -0.8, 'charge': 0, 'polarity': 1},
            'T': {'hydropathy': -0.7, 'charge': 0, 'polarity': 1},
            'W': {'hydropathy': -0.9, 'charge': 0, 'polarity': 0},
            'Y': {'hydropathy': -1.3, 'charge': 0, 'polarity': 1},
            'V': {'hydropathy': 4.2, 'charge': 0, 'polarity': 0}
        }

        hydropathy, charge, polarity = [], [], []
        for aa in sequence.upper():
            if aa in aa_properties:
                hydropathy.append(aa_properties[aa]['hydropathy'])
                charge.append(aa_properties[aa]['charge'])
                polarity.append(aa_properties[aa]['polarity'])

        features = [
            np.mean(hydropathy) if hydropathy else 0,
            np.std(hydropathy) if hydropathy else 0,
            np.mean(charge) if charge else 0,
            np.sum(charge) if charge else 0,
            np.mean(polarity) if polarity else 0,
            (sequence.upper().count('R') + sequence.upper().count('K') + sequence.upper().count('H')) / len(sequence),
            (sequence.upper().count('D') + sequence.upper().count('E')) / len(sequence),
        ]

        return features

    def _extract_composition_features(self, sequence):
        """提取氨基酸组成特征"""
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        composition = [sequence.upper().count(aa) / len(sequence) for aa in aa_list]
        return composition