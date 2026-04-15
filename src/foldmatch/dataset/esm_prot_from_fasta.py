import logging

import pandas as pd
from esm.sdk.api import ESMProtein
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def parse_fasta(fasta_file):
    sequences = []
    current_name = None
    current_seq_lines = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name is not None and current_seq_lines:
                    sequences.append((current_name, ''.join(current_seq_lines)))
                current_name = line[1:].split()[0]
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

    if current_name is not None and current_seq_lines:
        sequences.append((current_name, ''.join(current_seq_lines)))

    return sequences


class EsmProtFromFasta(Dataset):

    NAME_ATTR = 'name'
    SEQUENCE_ATTR = 'sequence'

    COLUMNS = [NAME_ATTR, SEQUENCE_ATTR]

    def __init__(
        self,
        fasta_file,
        min_res_n=0
    ):
        super().__init__()
        sequences = parse_fasta(fasta_file)
        self.data = pd.DataFrame(sequences, columns=EsmProtFromFasta.COLUMNS)
        if min_res_n > 0:
            self.data = self.data[self.data[EsmProtFromFasta.SEQUENCE_ATTR].str.len() >= min_res_n]
            self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data.iloc[idx][EsmProtFromFasta.NAME_ATTR]
        sequence = self.data.iloc[idx][EsmProtFromFasta.SEQUENCE_ATTR]
        return ESMProtein(sequence=sequence), name
