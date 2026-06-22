#!/usr/bin/env python3

import os
import json
import glob
import argparse

import numpy as np
import SimpleITK as sitk

from nnlandmark.dataset_conversion.generate_dataset_json import (
    generate_dataset_json
)

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# Flatten STS landmark JSON
# ============================================================

def extract_flat_landmarks(json_data):
    """
    Convert STS JSON structure into flat landmark dictionary.

    Example:

    input:
        006_upper_FDI_27:
            incisal:[x,y,z]

    output:
        upper_FDI_27_incisal:
        {
            coords:[x,y,z],
            arch:"upper"
        }

    """

    flat_landmarks = {}
    for tooth_key, features in json_data.items():
        parts = tooth_key.split("_")

        if "upper" in parts:
            arch = "upper"
        elif "lower" in parts:
            arch = "lower"
        else:
            arch = None

        # remove patient ID
        tooth_label = "_".join(parts[1:])

        for feat_name, feat_value in features.items():

            # ----------------------------
            # basePlane
            # ----------------------------

            if feat_name == "basePlane":
                for axis_name, axis_value in feat_value.items():
                    landmark_name = (
                        f"{tooth_label}_"
                        f"basePlane_"
                        f"{axis_name}"
                    )

                    flat_landmarks[landmark_name] = {
                        "coords": axis_value,
                        "arch": arch
                    }

            # ----------------------------
            # multiple points
            # cusps / planar
            # ----------------------------

            elif (
                isinstance(feat_value, list)
                and len(feat_value) > 0
                and isinstance(feat_value[0], list)
            ):
                for idx, point in enumerate(feat_value):
                    landmark_name = (
                        f"{tooth_label}_"
                        f"{feat_name}_"
                        f"{idx}"
                    )

                    flat_landmarks[landmark_name] = {
                        "coords": point,
                        "arch": arch
                    }

            # ----------------------------
            # single point
            # ----------------------------

            else:
                landmark_name = (
                    f"{tooth_label}_"
                    f"{feat_name}"
                )

                flat_landmarks[landmark_name] = {
                    "coords": feat_value,
                    "arch": arch
                }


    return flat_landmarks

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Create nnLandmark nnU-Net dataset from STS2026"
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=str
    )

    parser.add_argument(
        "-id",
        "--dataset_id",
        type=int,
        default=733
    )

    args = parser.parse_args()

    # ========================================================
    # Paths
    # ========================================================

    nnLM_raw = (
        os.environ.get("nnLM_raw")
        or "./nnLM_raw"
    )


    dataset_name = (
        f"Dataset{args.dataset_id:03d}_STS2026"
    )


    output_dir = os.path.join(
        nnLM_raw,
        dataset_name
    )


    train_root = os.path.join(
        args.input_dir,
        "Train-Unlabeled"
    )


    test_root = os.path.join(
        args.input_dir,
        "Validation"
    )



    imagesTr = os.path.join(
        output_dir,
        "imagesTr"
    )

    labelsTr = os.path.join(
        output_dir,
        "labelsTr"
    )

    imagesTs = os.path.join(
        output_dir,
        "imagesTs"
    )


    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(imagesTs)

    # ========================================================
    # Containers
    # ========================================================

    train_cases = []
    test_cases = []
    spacing_dict = {}
    all_landmarks_voxel = {}
    landmark_metadata = {}
    global_landmarks = set()
    overlap_counter = 0
    overlap_log = []

    # ========================================================
    # Phase 1
    # Collect landmark classes
    # ========================================================

    print("\nCollecting landmark classes...")

    train_paths = [
        p for p in glob.glob(os.path.join(train_root, "*")) if os.path.isdir(p)
    ]

    test_paths = [
        p for p in glob.glob(os.path.join(test_root, "*")) if os.path.isdir(p)
    ]

    for patient_folder in train_paths:
        json_files = glob.glob(os.path.join(patient_folder,"*.json"))

        if not json_files:
            continue

        with open(json_files[0], "r") as f:
            data = json.load(f)

        flat = extract_flat_landmarks(data)

        global_landmarks.update(
            flat.keys()
        )


    sorted_landmarks = sorted(
        list(global_landmarks)
    )

    landmark_to_id = {
        name: idx + 1
        for idx, name in enumerate(sorted_landmarks)
    }

    print(
        "Number of landmark classes:",
        len(landmark_to_id)
    )

    for k,v in list(landmark_to_id.items())[:20]:
        print(v, k)

    # ========================================================
    # Phase 2
    # Convert CBCT + generate landmark labels
    # ========================================================

    print("\nProcessing CBCT volumes...")

    for split_name, patient_paths in {
        "train": train_paths,
        "test": test_paths
    }.items():

        print("\nSplit:", split_name)

        for patient_folder in patient_paths:
            patient_id = os.path.basename(patient_folder)

            print("Processing:", patient_id)

            # ------------------------------------------------
            # CBCT
            # ------------------------------------------------

            nii_files = glob.glob(os.path.join(patient_folder, "*.nii.gz"))

            if not nii_files:
                print("Missing CBCT:", patient_id)
                continue


            cbct_path = nii_files[0]
            cbct_img = sitk.ReadImage(cbct_path)

            # ------------------------------------------------
            # Test images
            # ------------------------------------------------

            if split_name == "test":
                sitk.WriteImage(
                    cbct_img,
                    os.path.join(
                        imagesTs,
                        f"{patient_id}_0000.nii.gz"
                    )
                )

                test_cases.append(patient_id)
                continue

            # =================================================
            # Training
            # =================================================

            json_files = glob.glob(os.path.join(patient_folder,"*.json"))

            if not json_files:
                print("Missing landmark JSON:",patient_id)
                continue

            with open(json_files[0],"r") as f:
                landmark_json = json.load(f)

            flat_landmarks = extract_flat_landmarks(landmark_json)

            # ------------------------------------------------
            # Load transformations
            # ------------------------------------------------

            upper_matrix_path = os.path.join(
                patient_folder,
                "from_refframe_to_cbct_upper.npy"
            )

            lower_matrix_path = os.path.join(
                patient_folder,
                "from_refframe_to_cbct_lower.npy"
            )

            if not os.path.exists(upper_matrix_path):
                print("Missing upper transformation:", patient_id)
                continue

            if not os.path.exists(lower_matrix_path):
                print("Missing lower transformation:", patient_id)
                continue

            T_upper = np.load(
                upper_matrix_path
            )

            T_lower = np.load(
                lower_matrix_path
            )

            if T_upper.shape != (4,4):
                raise RuntimeError(
                    f"Wrong upper matrix shape {patient_id}"
                )

            if T_lower.shape != (4,4):
                raise RuntimeError(
                    f"Wrong lower matrix shape {patient_id}"
                )

            # ------------------------------------------------
            # Metadata spacing
            # ------------------------------------------------

            spacing_dict[patient_id] = {
                "image_spacing":
                    list(
                        cbct_img.GetSpacing()
                    ),
                "annotation_spacing":
                    None
            }

            # ------------------------------------------------
            # Empty label image
            # ------------------------------------------------

            label_img = sitk.Image(
                cbct_img.GetSize(),
                sitk.sitkUInt16
            )

            label_img.CopyInformation(
                cbct_img
            )

            label_array = sitk.GetArrayFromImage(
                label_img
            )

            case_voxel_landmarks = {}
            case_metadata = {}

            # =================================================
            # Each landmark
            # =================================================

            for landmark_name, landmark_info in flat_landmarks.items():
                if landmark_name not in landmark_to_id:
                    continue

                coords = landmark_info["coords"]
                arch = landmark_info["arch"]
                class_id = landmark_to_id[
                    landmark_name
                ]

                # ---------------------------------------------
                # IOS homogeneous coordinate
                # ---------------------------------------------

                p_ios = np.array(
                    [
                        coords[0],
                        coords[1],
                        coords[2],
                        1.0
                    ]
                )

                # ---------------------------------------------
                # Apply correct transformation
                # ---------------------------------------------

                if arch == "upper":
                    p_cbct = (T_upper @ p_ios)[:3]

                elif arch == "lower":
                    p_cbct = (T_lower @ p_ios)[:3]

                else:
                    print("Unknown arch:", landmark_name)
                    continue

                # ---------------------------------------------
                # Physical CBCT -> voxel
                # ---------------------------------------------

                try:
                    voxel_index = (
                        cbct_img
                        .TransformPhysicalPointToIndex(
                            tuple(p_cbct)
                        )
                    )

                except Exception as e:
                    print("Mapping error:", landmark_name, e)
                    continue

                x,y,z = voxel_index
                size_x, size_y, size_z = (cbct_img.GetSize())

                if not (0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z):
                    print("Landmark outside volume:", patient_id, landmark_name)
                    continue

                # =================================================
                # Save landmark coordinates
                # =================================================

                case_voxel_landmarks[landmark_name] = [
                    int(x),
                    int(y),
                    int(z)
                ]

                case_metadata[landmark_name] = {
                    "class_id": int(class_id),
                    "arch": arch,
                    "physical_cbct":
                        [
                            float(v)
                            for v in p_cbct
                        ],
                    "voxel_cbct":
                        [
                            int(x),
                            int(y),
                            int(z)
                        ]
                }

                # =================================================
                # Create 3x3x3 label cube
                # =================================================

                for dz in [-1,0,1]:
                    for dy in [-1,0,1]:
                        for dx in [-1,0,1]:

                            nx = x + dx
                            ny = y + dy
                            nz = z + dz

                            if (0 <= nx < label_array.shape[2] and 0 <= ny < label_array.shape[1] and 0 <= nz < label_array.shape[0]):
                                previous = label_array[
                                    nz,
                                    ny,
                                    nx
                                ]

                                if previous == 0:
                                    label_array[nz,ny,nx] = class_id
                                elif previous == class_id:
                                    pass
                                else:
                                    overlap_counter +=1
                                    overlap_log.append({
                                        "patient_id": patient_id,
                                        "existing_class": int(previous),
                                        "new_class": int(class_id),
                                        "voxel": [
                                            int(nx),
                                            int(ny),
                                            int(nz)
                                        ]
                                    })

            # ------------------------------------------------
            # Save metadata
            # ------------------------------------------------

            all_landmarks_voxel[patient_id] = (
                case_voxel_landmarks
            )

            landmark_metadata[patient_id] = (
                case_metadata
            )

            # ------------------------------------------------
            # Save images
            # ------------------------------------------------

            final_label_img = sitk.GetImageFromArray(
                label_array
            )

            final_label_img.CopyInformation(
                cbct_img
            )

            sitk.WriteImage(cbct_img, os.path.join(imagesTr, f"{patient_id}_0000.nii.gz"))

            sitk.WriteImage(final_label_img, os.path.join(labelsTr, f"{patient_id}.nii.gz"))

            train_cases.append(
                patient_id
            )

    # ========================================================
    # Phase 3
    # Save JSON metadata
    # ========================================================

    print("\nSaving metadata...")

    save_json(all_landmarks_voxel, os.path.join(output_dir, "all_landmarks_voxel.json"))
    save_json(landmark_metadata, os.path.join(output_dir, "landmark_metadata.json"))
    save_json(overlap_log, os.path.join(output_dir, "landmark_overlap_log.json"))
    save_json(spacing_dict, os.path.join(output_dir, "spacing.json"))

    # ========================================================
    # Split JSON
    # ========================================================

    split_dir = os.path.join(
        output_dir,
        "All_split_jsons"
    )

    maybe_mkdir_p(
        split_dir
    )

    split_json = {
        "imagesTr": sorted(train_cases),
        "imagesTs": sorted(test_cases),
        "all": sorted(list(set(train_cases) | set(test_cases)))
    }

    save_json(split_json, os.path.join(split_dir, f"split_{dataset_name}.json"))

    # ========================================================
    # Generate dataset.json using nnU-Net function
    # ========================================================

    labels = {
        "background":0,
        **landmark_to_id
    }

    generate_dataset_json(
        output_folder=output_dir,
        channel_names={
            0:"CBCT"
        },
        labels=labels,
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        dataset_name=dataset_name,
        description="STS2026 CBCT dental landmark localization dataset",
        converted_by="AimageLab-zip"
    )



    print("\n==============================")
    print(" DATASET CREATED SUCCESSFULLY ")
    print("==============================")

    print(
        "Training:",
        len(train_cases)
    )

    print(
        "Testing:",
        len(test_cases)
    )

    print(
        "Landmark classes:",
        len(landmark_to_id)
    )

    print(
    "\nTotal landmark overlaps:",
    overlap_counter
    )

    print(
        "Output:",
        output_dir
    )

if __name__ == "__main__":
    main()