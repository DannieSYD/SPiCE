import torch
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from itertools import groupby
from collections import defaultdict
from torch_geometric.data import extract_zip

from loaders.utils import mol_to_data_obj
from loaders.ensemble import EnsembleDataset, EnsembleMultiPartDatasetV2


class CommonEE:
    @property
    def descriptors(self):
        return ['de']

    @property
    def excluded_ids(self):
        return ['atrop-merg-enamide-phe-bn-h-B_R10333']

    @property
    def raw_file_names(self):
        return 'EE.zip'

    @property
    def num_molecules(self):
        return self.y.shape[0]

    @property
    def num_parts(self):
        return 2

    def _process_molecules(self, return_molecule_lists=False):
        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        label_file = raw_file.replace('.zip', '.csv')
        labels = pd.read_csv(label_file)

        raw_file = raw_file.replace('.zip', '.sdf')
        mols = defaultdict(list)
        with Chem.SDMolSupplier(raw_file, removeHs=False) as suppl:
            for mol in tqdm(suppl):
                data = mol_to_data_obj(mol)

                data.energy = float(mol.GetProp('energy'))
                data.smiles = mol.GetProp('smiles')
                data.substrate_id = mol.GetProp('substrate_id')
                data.ligand_id = mol.GetProp('ligand_id')
                data.part_id = 1 if mol.GetProp('config_id') == '1' else 0
                data.id = mol.GetProp('id')
                data.name = data.substrate_id + '_' + data.ligand_id
                data.mol = mol

                if data.name in self.excluded_ids:
                    continue
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                mols[data.name].append(data)

        cursor = 0
        ys = []

        if return_molecule_lists:
            data_list = [[] for _ in range(self.num_parts)]
        else:
            data_list = []
        for name, mol_list in tqdm(mols.items()):
            y = labels[labels['MergeID'] == name][self.descriptors[0]].values[0] / 100.0
            ys.append(y)
            grouped_mol_list = [list(g) for k, g in groupby(mol_list, key=lambda x: x.part_id)]
            for part_id, mol_list_per_part in enumerate(grouped_mol_list):
                grouped_mol_list[part_id] = sorted(mol_list_per_part, key=lambda x: x.energy)
                if self.max_num_conformers is not None:
                    grouped_mol_list[part_id] = grouped_mol_list[part_id][:self.max_num_conformers]
                for mol in grouped_mol_list[part_id]:
                    mol.molecule_idx = cursor
                if return_molecule_lists:
                    data_list[part_id].append(grouped_mol_list[part_id])
                else:
                    data_list.extend(grouped_mol_list[part_id])
            cursor += 1
        ys = torch.Tensor(ys).unsqueeze(1)
        return data_list, ys


class EEV2(CommonEE, EnsembleMultiPartDatasetV2):
    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return 'EEV2_processed.pt' if self.max_num_conformers is None \
            else f'EEV2_{self.max_num_conformers}_processed.pt'

    @property
    def num_molecules(self):
        return self.y.shape[0]

    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        molecule_lists, y = self._process_molecules(return_molecule_lists=True)
        torch.save((molecule_lists, y), self.processed_paths[0])


class EE_2D(CommonEE, EnsembleDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    @property
    def processed_file_names(self):
        return 'EE_2D_processed.pt'

    @staticmethod
    def process_molecule(mol, id, ligand_id, substrate_id, part_id, name):
        data = mol_to_data_obj(mol)
        data.id = id
        data.smiles = Chem.MolToSmiles(mol)
        data.ligand_id = ligand_id
        data.substrate_id = substrate_id
        data.part_id = part_id
        data.name = name
        return data

    def process(self):
        data_list = []

        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        label_file = raw_file.replace('.zip', '.csv')
        labels = pd.read_csv(label_file)

        raw_file = raw_file.replace('.zip', '.sdf')
        mols = {}
        with Chem.SDMolSupplier(raw_file, removeHs=False) as suppl:
            for mol in tqdm(suppl):
                id = mol.GetProp('id')
                substrate_id = mol.GetProp('substrate_id')
                ligand_id = mol.GetProp('ligand_id')
                name = '_'.join([substrate_id, ligand_id])

                if name in mols or name in self.excluded_ids:
                    continue

                frags = sorted(Chem.GetMolFrags(mol, asMols=True), key=lambda x: x.GetNumAtoms())
                data_0 = self.process_molecule(frags[0], id, ligand_id, substrate_id, 0, name)
                data_1 = self.process_molecule(frags[1], id, ligand_id, substrate_id, 1, name)

                if self.pre_filter is not None and not all(map(self.pre_filter, [data_0, data_1])):
                    continue
                if self.pre_transform is not None:
                    data_0, data_1 = map(self.pre_transform, [data_0, data_1])

                mols[name] = (data_0, data_1)

        cursor = 0
        ys = []
        for name, mol_list in tqdm(mols.items()):
            y = labels[labels['MergeID'] == name][self.descriptors[0]].values[0] / 100.0
            ys.append(y)
            for mol in mol_list:
                mol.molecule_idx = cursor
                data_list.append(mol)
            cursor += 1
        ys = torch.Tensor(ys).unsqueeze(1)

        data, slices = self.collate(data_list)
        torch.save((data, slices, ys), self.processed_paths[0])


class EE(CommonEE, EnsembleDataset):
    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    @property
    def processed_file_names(self):
        return 'EE_processed.pt' if self.max_num_conformers is None \
            else f'EE_{self.max_num_conformers}_processed.pt'

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
