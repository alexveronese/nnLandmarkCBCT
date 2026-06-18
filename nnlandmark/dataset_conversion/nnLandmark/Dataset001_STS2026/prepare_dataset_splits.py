import os
import json
import glob
import argparse
import random
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json, maybe_mkdir_p

def extract_flat_landmarks(json_data):
    """
    Flattens the structured JSON input into a single-level dictionary:
    {'upper_FDI_27_incisal': [x, y, z], 'upper_FDI_27_cusps_0': [x, y, z], ...}
    It strips the patient ID prefix from the tooth key to keep labels universal.
    """
    flat_landmarks = {}
    for tooth_key, features in json_data.items():
        parts = tooth_key.split('_')
        tooth_label = "_".join(parts[1:]) if len(parts) > 1 else tooth_key
        
        for feat_key, feat_val in features.items():
            if feat_key == "basePlane":
                for axis_key, axis_val in feat_val.items():
                    flat_name = f"{tooth_label}_{feat_key}_{axis_key}"
                    flat_landmarks[flat_name] = axis_val
            elif isinstance(feat_val, list) and len(feat_val) > 0 and isinstance(feat_val[0], list):
                for idx, coord in enumerate(feat_val):
                    flat_name = f"{tooth_label}_{feat_key}_{idx}"
                    flat_landmarks[flat_name] = coord
            else:
                flat_name = f"{tooth_label}_{feat_key}"
                flat_landmarks[flat_name] = feat_val
                
    return flat_landmarks

def main():
    # ==========================================
    # CLI ARGUMENT PARSING
    # ==========================================
    parser = argparse.ArgumentParser(description="Randomly split raw dataset and convert into nnLandmark format.")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Path to the single root folder containing all patient subfolders.")
    parser.add_argument("-id", "--dataset_id", type=int, default=733,
                        help="Three-digit dataset ID for nnU-Net/nnLandmark (default: 733).")
    parser.add_argument("-ts", "--test_size", type=float, default=0.20,
                        help="Percentage of the dataset to allocate to the test set (default: 0.20 for 20%).")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed for reproducibility of the train/test split (default: 42).")
    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)

    # Define paths based on nnU-Net environment variables or local fallback
    nnunet_raw = os.environ.get('nnUNet_raw') or "/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw"
    dataset_name = f"Dataset{args.dataset_id:03d}_STS2026"
    output_dir = os.path.join(nnunet_raw, dataset_name)

    # Subfolders setup for both Train (Tr) and Test (Ts)
    folders = {
        "train": {
            "images": os.path.join(output_dir, "imagesTr"),
            "labels": os.path.join(output_dir, "labelsTr"),
            "case_ids": []
        },
        "test": {
            "images": os.path.join(output_dir, "imagesTs"),
            "labels": os.path.join(output_dir, "labelsTs"),
            "case_ids": []
        }
    }

    for split in folders.values():
        maybe_mkdir_p(split["images"])
        maybe_mkdir_p(split["labels"])

    # Global dictionaries for nnLandmark metadata
    spacing_dict = {}
    all_landmarks_voxel_dict = {}
    global_landmark_keys = set()

    # ==========================================
    # PHASE 1: SCANNING & RANDOM SPLITTING
    # ==========================================
    print("Phase 1: Scanning dataset and performing random train/test split...")
    all_patient_folders = [f for f in glob.glob(os.path.join(args.input_dir, "*")) if os.path.isdir(f)]
    
    # Filter folders to ensure they actually contain a JSON and a NIfTI file before splitting
    valid_patient_folders = []
    for p_folder in all_patient_folders:
        p_id = os.path.basename(p_folder)
        has_json = len(glob.glob(os.path.join(p_folder, "*.json"))) > 0
        has_nifti = len(glob.glob(os.path.join(p_folder, "*.nii.gz"))) > 0
        if has_json and has_nifti:
            valid_patient_folders.append(p_folder)
        else:
            print(f" -> [Skip] Missing required files (.json or .nii.gz) in folder {p_id}")

    total_patients = len(valid_patient_folders)
    if total_patients == 0:
        print("[!] Error: No valid patient folders found. Check your input directory.")
        return

    print(f"Found {total_patients} valid patients. Proceeding to shuffle and split...")
    
    # Shuffle randomly using the deterministic seed
    random.shuffle(valid_patient_folders)
    
    # Calculate partition index
    num_test = int(total_patients * args.test_size)
    test_paths = valid_patient_folders[:num_test]
    train_paths = valid_patient_folders[num_test:]

    valid_dataset_structure = {
        "train": train_paths,
        "test": test_paths
    }

    # First pass: collect all unique landmark names from all files to create an immutable global mapping
    print("Extracting global landmark registry keys...")
    for split_type, paths in valid_dataset_structure.items():
        for p_folder in paths:
            json_file = glob.glob(os.path.join(p_folder, "*.json"))[0]
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    flat_lms = extract_flat_landmarks(data)
                    global_landmark_keys.update(flat_lms.keys())
                except Exception as e:
                    print(f" -> [Error] Corrupted JSON for patient {os.path.basename(p_folder)}: {e}")

    sorted_keys = sorted(list(global_landmark_keys))
    landmark_to_id = {name: idx + 1 for idx, name in enumerate(sorted_keys)}
    print(f"Total unique landmarks found across entire dataset: {len(landmark_to_id)}")

    # ==========================================
    # PHASE 2: PROCESSING VOLUMES & GEOMETRY
    # ==========================================
    print("\nPhase 2: Processing geometric transformations and generating NIfTI files...")

    for split_type, patient_paths in valid_dataset_structure.items():
        print(f"\n--- Processing Split: {split_type.upper()} ({len(patient_paths)} patients) ---")
        img_target_dir = folders[split_type]["images"]
        lbl_target_dir = folders[split_type]["labels"]
        
        for p_folder in patient_paths:
            p_id = os.path.basename(p_folder)
            print(f"Processing Patient: {p_id}")
            
            cbct_path = glob.glob(os.path.join(p_folder, "*.nii.gz"))[0]
            json_path = glob.glob(os.path.join(p_folder, "*.json"))[0]
            
            # Robust matrix searching (matches 'matrix.txt', 'matrix.npy', etc.)
            matrix_files = (glob.glob(os.path.join(p_folder, "*matrix*.*")) or 
                            glob.glob(os.path.join(p_folder, "*.npy")) or 
                            glob.glob(os.path.join(p_folder, "*.txt")))
            if not matrix_files:
                print(f"  [!] Error: Transformation matrix not found for {p_id}. Skipping.")
                continue
            
            matrix_path = matrix_files[0]
            T = np.load(matrix_path) if matrix_path.endswith('.npy') else np.loadtxt(matrix_path)
                
            cbct_img = sitk.ReadImage(cbct_path)
            
            # Store image spacing metadata
            spacing_dict[p_id] = {
                "image_spacing": list(cbct_img.GetSpacing()),
                "annotation_spacing": None
            }
            
            label_img = sitk.Image(cbct_img.GetSize(), sitk.sitkUInt16)
            label_img.CopyInformation(cbct_img)
            label_array = sitk.GetArrayFromImage(label_img)
            
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            flat_lms = extract_flat_landmarks(json_data)
            
            patient_voxel_coords = {}
            
            for lm_name, coords in flat_lms.items():
                if lm_name not in landmark_to_id:
                    continue
                label_id = landmark_to_id[lm_name]
                p_ios = np.array([coords[0], coords[1], coords[2], 1.0])
                p_cbct_physical = (T @ p_ios)[:3]
                
                try:
                    voxel_idx = cbct_img.TransformPhysicalPointToIndex(p_cbct_physical)
                    x, y, z = voxel_idx
                    patient_voxel_coords[lm_name] = [int(x), int(y), int(z)]
                    
                    # Generate 3x3x3 cubic voxel patches
                    for dz in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                nz, ny, nx = z + dz, y + dy, x + dx
                                if 0 <= nz < label_array.shape[0] and 0 <= ny < label_array.shape[1] and 0 <= nx < label_array.shape[2]:
                                    label_array[nz, ny, nx] = label_id
                                    
                except Exception as e:
                    print(f"  [!] Could not map landmark {lm_name} inside CBCT volume for patient {p_id}: {e}")

            all_landmarks_voxel_dict[p_id] = patient_voxel_coords

            final_label_img = sitk.GetImageFromArray(label_array)
            final_label_img.CopyInformation(cbct_img)
            
            # Save images using ONLY the patient ID folder name as case ID (e.g., 003_0000.nii.gz)
            sitk.WriteImage(cbct_img, os.path.join(img_target_dir, f"{p_id}_0000.nii.gz"))
            sitk.WriteImage(final_label_img, os.path.join(lbl_target_dir, f"{p_id}.nii.gz"))
            folders[split_type]["case_ids"].append(p_id)

    # ==========================================
    # PHASE 3: GENERATING METADATA AND SPLIT JSON FILES
    # ==========================================
    print("\nPhase 3: Generating all metadata and split JSON files...")
    
    # 1. Standard nnU-Net dataset.json
    dataset_json = {
        "channel_names": {"0": "CBCT"},
        "labels": {"background": 0, **{name: idx for name, idx in landmark_to_id.items()}},
        "numTraining": len(folders["train"]["case_ids"]),
        "file_ending": ".nii.gz"
    }
    save_json(dataset_json, os.path.join(output_dir, "dataset.json"))
    
    # 2. nnLandmark specific spacing.json
    save_json(spacing_dict, os.path.join(output_dir, "spacing.json"))
    
    # 3. nnLandmark specific all_landmarks_voxel.json
    save_json(all_landmarks_voxel_dict, os.path.join(output_dir, "all_landmarks_voxel.json"))
    
    # 4. Integrated Split Manifest
    tr_ids = sorted(folders["train"]["case_ids"])
    ts_ids = sorted(folders["test"]["case_ids"])
    split_manifest = {
        "imagesTr": tr_ids,
        "imagesTs": ts_ids,
        "all": sorted(list(set(tr_ids) | set(ts_ids)))
    }
    
    split_json_dir = os.path.join(output_dir, "All_split_jsons")
    maybe_mkdir_p(split_json_dir)
    save_json(split_manifest, os.path.join(split_json_dir, f"split_{dataset_name}.json"))

    print(f"\n[DONE] Dataset '{dataset_name}' successfully built and split!")
    print(f"Allocated: {len(tr_ids)} to Training (imagesTr) | {len(ts_ids)} to Testing (imagesTs)")
    print(f"All files saved inside: {output_dir}")

if __name__ == "__main__":
    main()