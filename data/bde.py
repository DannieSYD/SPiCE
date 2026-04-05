import torch
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from collections import defaultdict
from torch_geometric.data import extract_zip

from loaders.utils import mol_to_data_obj
from loaders.ensemble import EnsembleDataset, EnsembleMultiPartDatasetV2


class CommonBDE:
    @property
    def descriptors(self):
        return ['BindingEnergy']

    @property
    def raw_file_names(self):
        return 'BDE.zip'

    @property
    def num_parts(self):
        return 2

    def _process_molecules(self, return_molecule_lists=False):
        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        label_file = raw_file.replace('.zip', '.txt')
        labels = pd.read_csv(
            label_file, sep='  ', header=0,
            names=['Name', self.descriptors[0]], engine='python')

        filenames = ['substrates', 'ligands']
        mols = dict()
        mols_count = [defaultdict(int), defaultdict(int)]
        for part_id, filename in enumerate(filenames):
            raw_file = label_file.replace('BDE.txt', f'{filename}.sdf')
            with Chem.SDMolSupplier(raw_file, removeHs=True) as suppl:
                for mol in tqdm(suppl):
                    data = mol_to_data_obj(mol)

                    data.smiles = Chem.MolToSmiles(mol)
                    data.mol = mol
                    data.name = mol.GetProp('Name')
                    data.id = mol.GetProp('Index')
                    data.part_id = 1 if filename == 'ligands' else 0

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if self.max_num_conformers is not None and mols_count[part_id][data.name] >= self.max_num_conformers:
                        continue
                    if data.name not in mols:
                        mols[data.name] = defaultdict(list)
                    mols[data.name][part_id].append(data)
                    mols_count[part_id][data.name] += 1

        if return_molecule_lists:
            data_list = [[] for _ in range(self.num_parts)]
        else:
            data_list = []
        cursor = 0
        ys = []

        for name, mol_list in mols.items():
            y = labels[labels['Name'] == name][self.descriptors[0]].values[0]
            ys.append(y)

            for part_id in range(self.num_parts):
                for mol in mol_list[part_id]:
                    mol.molecule_idx = cursor
                if return_molecule_lists:
                    data_list[part_id].append(mol_list[part_id])
                else:
                    data_list.extend(mol_list[part_id])
            cursor += 1
        ys = torch.Tensor(ys).unsqueeze(1)
        return data_list, ys


class BDEV2(CommonBDE, EnsembleMultiPartDatasetV2):
    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return 'BDEV2_processed.pt' if self.max_num_conformers is None \
            else f'BDEV2_{self.max_num_conformers}_processed.pt'

    @property
    def num_molecules(self):
        return self.y.shape[0]

    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        molecule_lists, y = self._process_molecules(return_molecule_lists=True)
        torch.save((molecule_lists, y), self.processed_paths[0])


class BDE(CommonBDE, EnsembleDataset):
    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    @property
    def processed_file_names(self):
        return 'BDE_processed.pt' if self.max_num_conformers is None \
            else f'BDE_{self.max_num_conformers}_processed.pt'

    @property
    def num_molecules(self):
        return self.y.shape[0]

    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        data_list, ys = self._process_molecules(return_molecule_lists=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices, ys), self.processed_paths[0])
