import random

import torch
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from collections import defaultdict
from torch_geometric.data import extract_zip

from loaders.utils import mol_to_data_obj
from loaders.ensemble import EnsembleDataset, EnsembleMultiPartDatasetV2


def select_random_subset(data_list, y, fraction=0.1, seed=8888):

    num_samples = int(len(data_list) * fraction)
    random.seed(seed)
    selected_indices = random.sample(range(len(data_list)), num_samples)
    selected_data_list = [data_list[i] for i in selected_indices]
    selected_y = y[selected_indices]

    return selected_data_list, selected_y


class CommonDrugs:
    @property
    def descriptors(self):
        return ['energy', 'ip', 'ea', 'chi']

    @property
    def raw_file_names(self):
        return 'Drugs.zip'

    @property
    def num_parts(self):
        return 1

    def _process_molecules(self, return_molecule_lists=False):
        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        raw_file = raw_file.replace('.zip', '.sdf')

        label_file = raw_file.replace('.sdf', '.csv')
        labels = pd.read_csv(label_file)

        data_list = []
        y = []
        cursor = 0

        with Chem.SDMolSupplier(raw_file, removeHs=False) as suppl:
            molecule_dict = defaultdict(list)
            for mol in tqdm(suppl):
                id_ = mol.GetProp('ID')
                name = mol.GetProp('_Name')
                smiles = mol.GetProp('smiles')

                data = mol_to_data_obj(mol)
                data.name = name
                data.id = id_
                data.smiles = smiles
                data.mol = mol
                data.y = [float(mol.GetProp(descriptor)) for descriptor in self.descriptors]
                data.y = torch.Tensor(data.y).unsqueeze(0)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                molecule_dict[name].append(data)

        for name, conformers in tqdm(molecule_dict.items()):
            if self.max_num_conformers is not None:
                conformers = sorted(conformers, key=lambda x: x.y[:, self.descriptors.index('energy')].item())
                conformers = conformers[:self.max_num_conformers]

            for conformer in conformers:
                conformer.molecule_idx = cursor
            cursor += 1

            # if return_molecule_lists:
            #     data_list.append(conformers)
            # else:
            #     data_list.extend(conformers)
            data_list.append(conformers)

            row = labels[labels['name'] == name]
            y.append(torch.Tensor([row[quantity].item() for quantity in self.descriptors]))

        if return_molecule_lists:
            data_list = data_list
        y = torch.stack(y, dim=0)

        datalist, y = select_random_subset(data_list, y, fraction=0.1)

        if not return_molecule_lists:
            cursor = 0
            flat_data_list = []
            for conformers in tqdm(datalist):
                for conformer in conformers:
                    conformer.molecule_idx = cursor
                cursor += 1
                flat_data_list.extend(conformers)
            datalist = flat_data_list
        else:
            datalist = [datalist]
        return datalist, y


class DrugsV2(CommonDrugs, EnsembleMultiPartDatasetV2):
    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return 'DrugsV2_processed.pt' if self.max_num_conformers is None \
            else f'DrugsV2_processed_{self.max_num_conformers}.pt'

    def process(self):
        molecule_lists, y = self._process_molecules(return_molecule_lists=True)
        torch.save((molecule_lists, y), self.processed_paths[0])


class Drugs(CommonDrugs, EnsembleDataset):
    descriptors = ['energy', 'ip', 'ea', 'chi']

    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return 'DrugsEnsemble_processed.pt' if self.max_num_conformers is None \
            else f'DrugsEnsemble_processed_{self.max_num_conformers}.pt'

    @property
    def num_molecules(self):
        return self.y.shape[0]

    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        data_list, y = self._process_molecules(return_molecule_lists=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices, y), self.processed_paths[0])
