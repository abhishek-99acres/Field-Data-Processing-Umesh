import pandas as pd
import os
import shutil
import re
import chardet
import json
import datetime
import ast
import requests
import re
import numpy as np


# print(df.columns)


# Column Mappings
column_names = {
    "projectdetails_sources": "Project Details Sources",
    "projectdetails_openarea": "Open Area",
    "projectdetails_unitcount": "Unit Count",
    "projectdetails_totalarea": "Total Area",
    "projectdetails_greenarea": "Green Area",
    "projectdetails_floorcount": "Floor Count",
    "projectdetails_towercount": "Tower Count",
    "projectdetails_possessionstatus": "Possession Status",
    # "projectdetails_city": "City",
    # "projectdetails_locality": "Locality",
    "projectdetails_buildername": "Builder Name",
    "projectdetails_projectname": "Project Name",
    "projectdetails_possessionstatus": "Possession Status",
    "projectdetails_uspdetails": "USP Details",
    "projectdetails_address": "Project Address",
    "phaseIdentifier": "Phase",
    "projectdetails_latitude": "Latitude",
    "projectdetails_longitude": "Longitude",
    "projectdetails_reraregno": "REAR Reg No",
    "Sources": "Sources",
    "RERAID": "RERAID",
    "ConstructionStatus": "Construction Status",
    "CompletionDate": "Completion Date",
    "LaunchDate": "Launch Date",
    "SaleableArea": "Saleable Area",
    "phaseIdentifier": "Phase Identifier",
}


def create_xid(row):
    if row["rescom"] in ["RESIDENTIAL", "Residential", "residential"]:
        return f"R{row['XID']}"
    else:
        return f"C{row['XID']}"


def generate_id(row):
    date = row["Visit Date"]
    if pd.isna(date):
        date = row["Modify Date"].date()
    if pd.isna(date):
        return f"{row['XID']}_MISSING_DATE"  # Fallback if both are NaT
    else:
        return f"{row['XID']}_{pd.to_datetime(date).strftime('%Y%m%d')}"


def parse_basic_details(text):
    details = {}
    for line in str(text).splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # only include if key is present in mapping
            if key in column_names and val not in ("", "null", "[]", None):
                details[column_names[key]] = val
    return details


def parse_phase_identifier(phase_identifier):
    if not isinstance(phase_identifier, str):
        return {"phaseIdentifier": "", "RERA_Number": ""}

    # Extract phase name (everything before _<number>_UC_ or _<number>_RTM_)
    phase_name_match = re.search(r"(.+?)_\d+_(?:UC_|RTM_)", phase_identifier)
    phase_name = phase_name_match.group(1).strip() if phase_name_match else ""

    # Extract project/RERA code (everything after UC_ or RTM_)
    rera_match = re.search(r"(?:RTM_|UC_)([A-Za-z0-9/_-]+)", phase_identifier)
    rera_number = rera_match.group(1) if rera_match else ""
    # rera_number = "" if rera_number.lower() == "null" else rera_number

    # Clean invalid/null values
    if not rera_number or rera_number.lower() == "null":
        return {"Phase Identifier": phase_name}

    return {"Phase Identifier": phase_name, "RERA Number": rera_number}


def parse_phase_blocks(text):
    # Split text into blocks starting with 'phaseIdentifier:'
    blocks = re.split(r"(?=phaseIdentifier:)", text.strip())
    result = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        data = {}
        for line in block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                if val.strip() == "null" or val.strip() == "" or val.strip() == "[]":
                    continue
                data[column_names[key.strip()]] = val.strip()

        # If phaseIdentifier exists, parse it
        if "Phase Identifier" in data:
            parsed = parse_phase_identifier(data["Phase Identifier"])
            for key, val in parsed.items():
                data[key] = val

        if data:  # only add if block has data
            result.append(data)

    return result


def parse_brochure(text):
    column_mapping = {
        "original": "Brochure Link",
        "Source": "Brochure Source",
        "phaseIdentifier": "Phase Identifier",
    }

    blocks = re.split(r"(?=phaseIdentifier:)", text.strip())
    result = []

    # print(blocks)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        details = {}  # move inside loop so it resets for each block
        for line in block.splitlines():  # ✅ use block, not text
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().replace("[", "").replace("]", "").replace("'", "")

                if key in column_mapping and val not in ("", "null", "[]", None):
                    details[column_mapping[key]] = val

        # Optional: parse phaseIdentifier if present
        if "Phase Identifier" in details:
            parsed = parse_phase_identifier(details["Phase Identifier"])
            details["Phase Identifier"] = parsed["Phase Identifier"]
            # details["RERA_Number"] = parsed["RERA_Number"]

        if details:
            result.append(details)

    return result


def parse_tower_details(text):

    column_mapping = {
        "phaseIdentifier": "Phase Identifier",
        "towerName": "Tower Name:",
        "totalFloorNo": "Total Floor No",
        "propertyType": "Property Type",
        "bhkConfig": "BHK Config",
        "NoofLifts:": "Lift Count",
        "minUnitsPerFloor": "Minimum Unit Per Floor Count",
        "maxUnitsPerFloor": "Maximum Unit Per Floor Count",
        "unitEntranceFacing": "Unit Entrance Facing",
        "unitViewFacing": "Unit View Facing",
        "towerOpenSide": "Tower Open Side",
        "source": "Tower Details Source",
    }

    result = {}
    blocks = re.split(r"(?=phaseIdentifier:)", text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        details = {}
        for line in block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()

                if key in column_mapping and val not in ("", "null", "[]", None):
                    details[column_mapping[key]] = val

        if "Phase Identifier" in details:
            parsed = parse_phase_identifier(details["Phase Identifier"])

            if parsed["Phase Identifier"] not in result:
                details.pop("Phase Identifier")
                result[parsed["Phase Identifier"]] = [details]
            else:
                details.pop("Phase Identifier")
                result[parsed["Phase Identifier"]].append(details)

            # result["Phase Identifier"] = parsed["phaseIdentifier"]

    return result


def parse_payment_plan(text):
    column_mapping = {
        "original": "Payment Document Link",
        "Source": "Payment Document Source",
        "paymentPlanType:": "Payment Plan Type",
        "phaseIdentifier": "Phase Identifier",
    }

    blocks = re.split(r"(?=phaseIdentifier:)", text.strip())
    result = []

    # print(blocks)

    details = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        for line in block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                # val = val.strip()
                val = val.strip().split("?", 1)[0]

                if key in column_mapping and val not in ("", "null", "[]", None):
                    details[column_mapping[key]] = val

        if "Phase Identifier" in details:
            parsed = parse_phase_identifier(details["Phase Identifier"])
            details["Phase Identifier"] = parsed["Phase Identifier"]

        if details:
            result.append(details)

    return result


def parse_oc_cc_certificate(text):
    column_mapping = {
        "original": "Certificate URL",
        "phaseIdentifier": "Phase Identifier",
        "source": "Certificate Source",
        "towerId": "Tower ID",
    }

    result = []
    blocks = re.split(r"(?=phaseIdentifier:)", text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        details = {}
        for line in block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().split("?", 1)[0]

                if key in column_mapping and val not in ("", "null", "[]", None):
                    details[column_mapping[key]] = val

        if "Phase Identifier" in details:
            parsed = parse_phase_identifier(details["Phase Identifier"])
            details["Phase Identifier"] = parsed["Phase Identifier"]

        if details:
            result.append(details)

    return result


def parse_options(text):
    column_mapping = {
        "areaUnit": "Options Area Unit",
        "bhk": "Options BHK",
        "builtupArea": "Options Builtup Area",
        "carpetArea": "Options Carpet Area",
        "comments": "Options Comments",
        "isInvalid": "Options Is Invalid",
        "isNew": "Options Is New",
        "original": "Options URL",
        "plotArea": "Options Plot Area",
        "propertyType": "Options Property Type",
        "superArea": "Options Super Area",
    }

    result = []
    blocks = re.split(r"(?=superArea:)", text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        details = {}
        for line in block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()

                if key in column_mapping and val != "":
                    details[column_mapping[key]] = val

        if details:
            result.append(details)

    return result


def parse_prices(text):

    column_mapping = {
        "comments": "Comments",
        "isallinclusive": "Is All Inclusive",
        "islaunchprice": "Is Launch Price",
        "original": "Price Document URL",
        "phaseidentifier": "Phase Identifier",
        "pricecategory": "Price Category",
        "source": "Price Source",
        "typeofprices": "Type of Prices",
        "visitoutcome": "Visit Outcome",
    }

    result = []
    blocks = re.split(r"(?=phaseIdentifier:)", text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        details = {}
        for line in block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().split("?", 1)[0]

                if key in column_mapping and val not in ("", "null", "[]", None):
                    details[column_mapping[key]] = val

        if "Phase Identifier" in details:
            parsed = parse_phase_identifier(details["Phase Identifier"])
            details["Phase Identifier"] = parsed["Phase Identifier"]

        if details:
            result.append(details)

    return result


def parse_urls(text):

    result = []

    for line in str(text).splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            # key = key.strip()
            val = val.strip()
            result.append(val)

    return result


def process_data(df):
    df.rename(columns={"visitdate": "Visit Date"}, inplace=True)

    df["XID"] = df.apply(create_xid, axis=1)

    df["Visit Date"] = pd.to_datetime(df["Visit Date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    df["Modify Date"] = pd.to_datetime(df["Modify Date"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    df["ID"] = df.apply(generate_id, axis=1)

    df.drop(columns=["User"], inplace=True)

    df["Basic Details"] = df["Basic Details"].astype(str).apply(parse_basic_details)

    df["Phase & Construction Status"] = (
        df["Phase & Construction Status"].astype(str).apply(parse_phase_blocks)
    )

    df.rename(
        columns={"Phase & Construction Status": "Phase And Construction Status"},
        inplace=True,
    )

    # df.drop(columns=["Phase & Construction Status"], inplace=True)

    df["Brochure"] = df["Brochure"].astype(str).apply(parse_brochure)

    df["Payment Plan"] = df["Payment Plan"].astype(str).apply(parse_payment_plan)

    df["Tower Details"] = df["Tower Details"].astype(str).apply(parse_tower_details)

    df["OC/CC Certificate"] = (
        df["OC/CC Certificate"].astype(str).apply(parse_oc_cc_certificate)
    )
    # df.to_excel("test.xlsx", index=False)

    df["Options Added"] = df["Options Added"].astype(str).apply(parse_options)
    # df.to_excel("test.xlsx", index=False)

    df["Prices"] = df["Prices"].astype(str).apply(parse_prices)

    df.rename(columns={"Project": "Project Images"}, inplace=True)
    df["Project Images"] = df["Project Images"].astype(str).apply(parse_urls)
    df["Locality.1"] = df["Locality.1"].astype(str).apply(parse_urls)
    df["Raw Video"] = df["Raw Video"].astype(str).apply(parse_urls)
    df["Video URL"] = df["Video URL"].astype(str).apply(parse_urls)

    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index("ID")))
    df = df[cols]
    # df_xl = df

    # df.to_excel(
    #     f"Processed_data_on_11_10_2025.xlsx",
    #     index=False,
    # )

    return df


#   #######################################################################


# 1️⃣ Helper functions
# ----------------------------
# def safe_eval(v):
#     """Safely convert Python dict/list strings to objects if possible."""
#     if isinstance(v, str) and v.strip().startswith(("{", "[")):
#         try:
#             return ast.literal_eval(v)
#         except Exception:
#             return v
#     return v


# def deep_diff(old, new, path=""):
#     """Recursively find differences between dict/list/scalar values."""
#     diffs = {}

#     if isinstance(old, dict) and isinstance(new, dict):
#         all_keys = set(old.keys()) | set(new.keys())
#         for k in all_keys:
#             sub_path = f"{path}.{k}" if path else k
#             diffs.update(deep_diff(old.get(k), new.get(k), sub_path))

#     elif isinstance(old, list) and isinstance(new, list):
#         max_len = max(len(old), len(new))
#         for i in range(max_len):
#             sub_path = f"{path}[{i}]"
#             val_old = old[i] if i < len(old) else None
#             val_new = new[i] if i < len(new) else None
#             diffs.update(deep_diff(val_old, val_new, sub_path))
#     else:
#         if old != new:
#             diffs[path] = {"old": old, "new": new}

#     return diffs


# # Post data (Insert new records in DB)
# def insert_data(data):
#     url = "http://172.16.3.229/NL_Upgrade/api/field_details_api.php"

#     print("\n🚀------------------------------------🚀\n")
#     print("Inserting data ...\n")

#     resp = requests.post(url, json=data)

#     print("Status Code : ", resp.status_code)
#     print("Response Text : ", resp.text)
#     print("Response : ", resp.json())

#     print("\nData Insertion completed\n")

#     print("\n🛑------------------------------------🛑\n")


# #   Put data (Update existing records in DB)
# def update_data(data):
#     url = "http://172.16.3.229/NL_Upgrade/api/field_details_api.php"

#     print("\n🚀------------------------------------🚀\n")
#     print("Updating data ...\n")

#     resp = requests.post(url, json=data)

#     print("Status Code : ", resp.status_code)
#     print("Response Text : ", resp.text)
#     print("Response : ", resp.json())

#     print("\nData Update completed\n")

#     print("\n🛑------------------------------------🛑\n")


# def get_table_data():
#     url = "http://172.16.3.229/NL_Upgrade/api/field_details_api.php?key=all&value=all"

#     print("\n🚀------------------------------------🚀\n")
#     print("Fetching data ...\n")

#     resp = requests.get(url)

#     if resp.status_code == 200:
#         print("Data fetched successfully")
#     else:
#         print("Failed to fetch data")

#     print("🛑----------------------------------🛑\n")

#     return resp.json()


# def compare_dataframes(df_db, df_xl):
#     # --- Normalize Columns ---
#     # df_xl.columns = [to_snake_case(col) for col in df_xl.columns]
#     key = "XID"

#     # --- Align Columns ---
#     common_cols = list(set(df_db.columns) & set(df_xl.columns))
#     common_cols.sort()

#     # --- Identify record categories ---
#     ids_db = set()
#     if not df_db.empty:
#         ids_db = set(df_db[key])

#     ids_xl = set(df_xl[key])

#     new_ids = ids_xl - ids_db  # only in Excel
#     deleted_ids = ids_db - ids_xl  # only in DB
#     existing_ids = ids_db & ids_xl  # in both

#     updated_records = []
#     new_records = []
#     final_records = []
#     updated_id = []

#     # --- Case 1: New records ---
#     for id_ in new_ids:
#         row = df_xl[df_xl[key] == id_].iloc[0].to_dict()
#         new_records.append(row)
#         final_records.append(row)

#     # --- Case 2: Deleted records (optional) ---
#     # You can choose to skip or log them
#     for id_ in deleted_ids:
#         row = df_db[df_db[key] == id_].iloc[0].to_dict()
#         final_records.append(row)

#     # --- Case 3: Existing IDs → compare field by field ---
#     for id_ in existing_ids:
#         old_row = df_db[df_db[key] == id_].iloc[0].to_dict()
#         new_row = df_xl[df_xl[key] == id_].iloc[0].to_dict()

#         diffs = {}
#         for col in common_cols:
#             old_val = safe_eval(old_row.get(col))
#             new_val = safe_eval(new_row.get(col))
#             if old_val != new_val:
#                 diff = deep_diff(old_val, new_val)
#                 if diff:
#                     diffs[col] = diff

#         if diffs:  # only if any change
#             updated_id.append(id_)
#             updated_records.append(new_row)
#             final_records.append(new_row)
#         else:
#             final_records.append(old_row)

#     # --- Build final DataFrames ---
#     df_final = pd.DataFrame(final_records)
#     updated_df = pd.DataFrame(updated_records)
#     new_df = pd.DataFrame(new_records)

#     # Add method for clarity
#     # updated_df["Status"] = "Updated"

#     # new_df = new_df.merge(updated_df, how="inner", on=key)

#     # Combine both dataframes vertically
#     merged_df = pd.concat([new_df, updated_df], ignore_index=True)

#     # --- Write results ---
#     df_final.to_excel("final_data.xlsx", index=False)
#     updated_df.to_excel("updated_records.xlsx", index=False)
#     new_df.to_excel("new_records.xlsx", index=False)
#     merged_df.to_excel("merged_df.xlsx", index=False)

#     return merged_df


def safe_eval(value):
    """Convert NA-like values to None for fair comparison."""
    if pd.isna(value) or str(value).strip().upper() in {"N/A", "NA", ""}:
        return None
    return value


def compare_dataframes(df_db, df_xl):
    key = "XID"

    # --- Align columns ---
    common_cols = sorted(list(set(df_db.columns) & set(df_xl.columns)))

    ids_db = set(df_db[key]) if not df_db.empty else set()
    ids_xl = set(df_xl[key])

    new_ids = ids_xl - ids_db  # only in Excel
    existing_ids = ids_db & ids_xl  # common IDs

    new_records = []
    updated_records = []

    # --- Case 1: New records (only in Excel) ---
    for id_ in new_ids:
        row = df_xl.loc[df_xl[key] == id_].iloc[0].to_dict()
        new_records.append(row)

    # --- Case 2: Existing IDs → compare field by field ---
    for id_ in existing_ids:
        old_row = df_db.loc[df_db[key] == id_].iloc[0].to_dict()
        new_row = df_xl.loc[df_xl[key] == id_].iloc[0].to_dict()

        updated = False
        for col in common_cols:
            old_val = safe_eval(old_row.get(col))
            new_val = safe_eval(new_row.get(col))
            if old_val != new_val:
                updated = True
                break

        if updated:
            updated_records.append(new_row)

    # --- Combine new and updated rows vertically ---
    merged_df = pd.concat(
        [pd.DataFrame(new_records), pd.DataFrame(updated_records)], ignore_index=True
    )

    # --- Optional: save to Excel ---
    # pd.DataFrame(new_records).to_excel("new_records.xlsx", index=False)
    # pd.DataFrame(updated_records).to_excel("updated_records.xlsx", index=False)
    # merged_df.to_excel("merged_df.xlsx", index=False)

    return merged_df


def main():

    new_data_path = "C:/Users/abhishek.k11/Desktop/Projects/Project-8 (Sonia M.)/src/Umesh/data/Field New Panel Data.xlsx"
    old_data_path = "C:/Users/abhishek.k11/Desktop/Projects/Project-8 (Sonia M.)/src/Umesh/incremental_data/Reference_DB_Data.xlsx"
    df = pd.read_excel(new_data_path, skiprows=1)
    df_db = pd.read_excel(
        old_data_path,
        keep_default_na=False,
    )
    # df = df.head(100)
    # df["Project Name"] = df.apply(
    #     lambda x: f"{x["Project Name"].strip()} Updated Today ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
    #     axis=1,
    # )

    df_xl = process_data(df)
    # df_db = pd.read_excel(old_data_path)

    # print(df_xl["Project Name"].head(10))

    def flatten_and_merge_columns(df, columns_to_flatten, dedup_key=None):
        """
        Flatten columns containing dicts or lists of dicts into common columns,
        merging all keys together, and deduplicating based on dedup_key.
        """
        flattened_rows = []

        for idx, row in df.iterrows():
            base_data = row.drop(columns=columns_to_flatten, errors="ignore").to_dict()
            merged_items = []

            for col in columns_to_flatten:
                cell = row.get(col)
                if isinstance(cell, dict):
                    merged_items.append(cell)
                elif isinstance(cell, list):
                    for item in cell:
                        if isinstance(item, dict):
                            merged_items.append(item)

            # # Deduplicate merged items based on dedup_key
            # seen_keys = set()
            # if merged_items:
            #     for item in merged_items:
            #         if dedup_key and dedup_key in item:
            #             key_val = item[dedup_key]
            #             if key_val in seen_keys:
            #                 continue
            #             seen_keys.add(key_val)
            #         flattened_rows.append({**base_data, **item})
            # else:
            #     flattened_rows.append(base_data)

            # ✅ No deduplication — keep every merged item
            if merged_items:
                for item in merged_items:
                    flattened_rows.append({**base_data, **item})
            else:
                flattened_rows.append(base_data)

        flat_df = pd.DataFrame(flattened_rows)
        flat_df = flat_df.loc[
            :, ~flat_df.columns.duplicated()
        ]  # remove duplicate columns
        return flat_df

    def expand_url_columns(df, url_columns):
        """
        Expand list-type URL columns into separate columns with counter suffix.
        """
        for col in url_columns:
            # Find max number of URLs in this column
            max_len = (
                df[col].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
            )

            # Create new columns: col_1, col_2, ...
            for i in range(max_len):
                new_col = f"{col}_{i+1}"
                df[new_col] = df[col].apply(
                    lambda x: x[i] if isinstance(x, list) and len(x) > i else None
                )

            # Drop original column
            df.drop(columns=[col], inplace=True)

        return df

    # Columns to flatten (these are your parsed dict/list columns)
    columns_to_flatten = [
        "Basic Details",
        "Phase And Construction Status",
        "Brochure",
        "Payment Plan",
        "Options Added",
        "OC/CC Certificate",
        "Prices",
    ]

    # Flatten and merge, deduplicating on 'Phase Identifier'
    df_flat = flatten_and_merge_columns(
        df_xl, columns_to_flatten, dedup_key="Phase Identifier"
    )

    print("Flattened DataFrame : ", df_flat.shape)
    # URL columns to expand
    url_columns = ["Project Images", "Video URL"]

    # Apply after flattening
    df_flat = expand_url_columns(df_flat, url_columns)
    df_flat.drop(
        columns=[
            "Basic Details",
            "Phase And Construction Status",
            "Brochure",
            "Payment Plan",
            "Prices",
            "Tower Details",
            "Prices",
            "Locality.1",
            "Raw Video",
            # "Video URL",
            "Amenity (In Brochure)",
            "Amenities Added",
            "Options (In Brochure)",
            "Additional Details",
            "Additional Comments",
            "Info Not Available",
            "Options Added",
            "OC/CC Certificate",
        ],
        inplace=True,
    )

    # # Save result
    # df_flat.to_excel("flattened_data.xlsx", index=False)
    # # selected_columns = df_flat[
    # #     df_flat["XID", "Total Area", "Open Area", "Floor Count", "Tower Count"]
    # # ]

    # Step 1: Select columns
    selected_columns = df_flat[
        [
            "XID",
            "Total Area",
            "Open Area",
            "Floor Count",
            "Tower Count",
            "Brochure Link",
        ]
    ].copy()

    # Step 2: Convert numeric columns to numbers
    num_cols = ["Total Area", "Open Area", "Floor Count", "Tower Count"]
    selected_columns[num_cols] = selected_columns[num_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Step 3: Group by XID and handle duplicates
    def combine_links(group):
        # Combine unique brochure links, drop blanks
        links = group["Brochure Link"].dropna().unique().tolist()
        combined_link = ", ".join(links) if links else np.nan

        # Keep the first non-null value for numeric columns
        numeric_data = group[num_cols].bfill().iloc[0]

        return pd.Series(
            [group["XID"].iloc[0], *numeric_data, combined_link],
            index=["XID", *num_cols, "Brochure Link"],
        )

    cleaned_df = selected_columns.groupby("XID", as_index=False).apply(combine_links)

    # Step 4: Replace NaN with "NA" for export
    cleaned_df = cleaned_df.replace({np.nan: "NA"})

    new_data = compare_dataframes(df_db=df_db, df_xl=cleaned_df)

    # Identify columns other than the ID
    non_id_cols = [c for c in new_data.columns if c != "XID"]

    # Drop rows where all non-ID columns are "NA"
    new_data = new_data.loc[
        ~(new_data[non_id_cols].applymap(lambda x: x) == "NA").all(axis=1)
    ]

    # Step 5: Export to Excel
    updated_reference_db = pd.concat([df_db, new_data], ignore_index=True)
    updated_reference_db = updated_reference_db.drop_duplicates(keep="last")
    updated_reference_db.to_excel(old_data_path, index=False)

    today = datetime.datetime.now().strftime("%Y-%m-%d")  # e.g. '2025-11-11'

    to_save = os.path.join(
        os.path.split(old_data_path)[0],
        f"Incremental_Data_{today}.xlsx",
    )
    new_data.to_excel(to_save, index=False)

    # post_data_json, put_data_json = compare_dataframes(df_db=df_db, df_xl=df_xl)
    # print("\n---------------------------------------------------------")
    # print("Post Data\n")
    # print(post_data_json)
    # print("\n---------------------------------------------------------\n")

    # print("Put Data\n")
    # print(put_data_json)
    # print("\n---------------------------------------------------------\n")

    print("Thanks for your patience")
    exit()


if __name__ == "__main__":
    main()
