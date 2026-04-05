import pickle

import torch
from rdkit import Chem
from torch_geometric.data import extract_zip
from tqdm import tqdm

from loaders.ensemble import EnsembleDataset, EnsembleMultiPartDatasetV2
from loaders.utils import mol_to_data_obj


class CommonKraken:
    @property
    def descriptors(self):
        return [
            "E_oxidation",
            "E_reduction",
            "E_solv_cds",
            "E_solv_elstat",
            "E_solv_total",
            "Pint_P_int",
            "Pint_P_max",
            "Pint_P_min",
            "Pint_dP",
            "dipolemoment",
            "efg_amp_P",
            "efgtens_xx_P",
            "efgtens_yy_P",
            "efgtens_zz_P",
            "fmo_e_homo",
            "fmo_e_lumo",
            "fmo_eta",
            "fmo_mu",
            "fmo_omega",
            "fukui_m",
            "fukui_p",
            "nbo_P",
            "nbo_P_ra",
            "nbo_P_rc",
            "nbo_bd_e_avg",
            "nbo_bd_e_max",
            "nbo_bd_occ_avg",
            "nbo_bd_occ_min",
            "nbo_bds_e_avg",
            "nbo_bds_e_min",
            "nbo_bds_occ_avg",
            "nbo_bds_occ_max",
            "nbo_lp_P_e",
            "nbo_lp_P_occ",
            "nbo_lp_P_percent_s",
            "nmr_P",
            "nmrtens_sxx_P",
            "nmrtens_syy_P",
            "nmrtens_szz_P",
            "nuesp_P",
            "qpole_amp",
            "qpoletens_xx",
            "qpoletens_yy",
            "qpoletens_zz",
            "somo_ra",
            "somo_rc",
            "sphericity",
            "spindens_P_ra",
            "spindens_P_rc",
            "sterimol_B1",
            "sterimol_B5",
            "sterimol_L",
            "sterimol_burB1",
            "sterimol_burB5",
            "sterimol_burL",
            "surface_area",
            "vbur_far_vbur",
            "vbur_far_vtot",
            "vbur_max_delta_qvbur",
            "vbur_max_delta_qvtot",
            "vbur_near_vbur",
            "vbur_near_vtot",
            "vbur_ovbur_max",
            "vbur_ovbur_min",
            "vbur_ovtot_max",
            "vbur_ovtot_min",
            "vbur_qvbur_max",
            "vbur_qvbur_min",
            "vbur_qvtot_max",
            "vbur_qvtot_min",
            "vbur_ratio_vbur_vtot",
            "vbur_vbur",
            "vbur_vtot",
            "vmin_r",
            "vmin_vmin",
            "volume",
        ]
        # Not all ligands have descriptors 'nbo_delta_lp_P_bds', 'pyr_P', 'pyr_alpha'

    @property
    def target_descriptors(self):
        return [
            "dipolemoment",
            "qpoletens_xx",
            "qpoletens_yy",
            "qpoletens_zz",
            "qpole_amp",
            "sterimol_B5",
            "sterimol_L",
            "sterimol_burB5",
            "sterimol_burL",
        ]

    @property
    def raw_file_names(self):
        return "Kraken.zip"

    @property
    def num_parts(self):
        return 1

    def _process_molecules(self, return_molecule_lists=False):
        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        raw_file = raw_file.replace(".zip", ".pickle")
        with open(raw_file, "rb") as f:
            kraken = pickle.load(f)

        data_list = []
        y = []
        ligand_ids = list(kraken.keys())
        cursor = 0
        for ligand_id in tqdm(ligand_ids):
            smiles, boltz_avg_properties, conformer_dict = kraken[ligand_id]
            conformer_ids = list(conformer_dict.keys())
            conformer_ids = sorted(
                conformer_ids, key=lambda x: conformer_dict[x][1], reverse=True
            )
            if self.max_num_conformers is not None:
                # sort conformers by boltzmann weight and take the lowest energy conformers
                conformer_ids = conformer_ids[: self.max_num_conformers]

            if return_molecule_lists:
                molecule_list = []
            for conformer_id in conformer_ids:
                mol_sdf, boltz_weight, conformer_properties = conformer_dict[
                    conformer_id
                ]
                mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False)

                data = mol_to_data_obj(mol)
                data.name = f"mol{int(ligand_id)}"
                data.id = f"{data.name}_{conformer_id}"
                data.smiles = smiles
                data.mol = mol
                data.y = torch.Tensor(
                    [
                        conformer_properties[descriptor]
                        for descriptor in self.descriptors
                    ]
                ).unsqueeze(0)
                data.molecule_idx = cursor

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                if return_molecule_lists:
                    molecule_list.append(data)
                else:
                    data_list.append(data)
            cursor += 1

            y.append(
                torch.Tensor(
                    [
                        boltz_avg_properties[descriptor]
                        for descriptor in self.descriptors
                    ]
                )
            )
            if return_molecule_lists:
                data_list.append(molecule_list)
        if return_molecule_lists:
            data_list = [data_list]
        return data_list, torch.stack(y, dim=0)


class KrakenV2(CommonKraken, EnsembleMultiPartDatasetV2):
    def __init__(
        self,
        root,
        max_num_conformers=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return (
            "KrakenV2_processed.pt"
            if self.max_num_conformers is None
            else f"KrakenV2_{self.max_num_conformers}_processed.pt"
        )

    def process(self):
        molecule_lists, y = self._process_molecules(return_molecule_lists=True)
        torch.save((molecule_lists, y), self.processed_paths[0])


class Kraken(CommonKraken, EnsembleDataset):
    def __init__(
        self,
        root,
        max_num_conformers=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    @property
    def processed_file_names(self):
        return (
            "Kraken_processed.pt"
            if self.max_num_conformers is None
            else f"Kraken_{self.max_num_conformers}_processed.pt"
        )

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
